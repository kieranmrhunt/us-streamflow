import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt, timedelta as td

from sklearn.metrics import r2_score
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler

def data_split(predictors, n_timestamp):
	X = []
	for i in range(len(predictors)):
		end_ix = i + n_timestamp
		if end_ix > len(predictors)-1:
			break
		seq_x = predictors[i:end_ix+1]
		X.append(seq_x)
	return np.array(X)


compute_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/compute/"



station_codes= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']

for start_date in [dt(2020,9,1)+td(days=i) for i in range(395)]:
	
	ndays = 10 #length of forecast
	datelist = [start_date+td(hours=18) + i*td(hours=6) for i in range(0,4*ndays)]

	handle = 'TC+KHGMEC_lstm'
	fname = start_date.strftime('hindcasts/lstm_%Y%m%dT%H.csv')

	if os.path.isfile(fname):
		continue

	f = open(fname, 'w+')
	f.write('DateTime,LocationID,ForecastTime,VendorID,Value,Units\n')

	fcast_dstring = start_date.strftime('%Y-%m-%dT%H')
	f_dstring = start_date.strftime('%Y%m%dT%H')

	gt_df = pd.read_csv(compute_dir+'recent_obs.csv')
	#print(gt_df)




	for station_code in station_codes:
		print(start_date, station_code)
		
		obs_df = pd.read_csv(compute_dir+"catchment-data/{}.csv".format(station_code), parse_dates=[0,],
		             date_parser=lambda t:pd.to_datetime(str(t), format='%Y-%m-%dT%H'))
		inflow = obs_df['inflow_obs'].values
		predictand = np.array(inflow)[:,None]
		obs_glofas_c = obs_df['glofas_c'].values
		obs_glofas_s = obs_df['glofas_s'].values
		obs_average = obs_df['inflow_avg'].values

		
		pretrained_scores = pd.read_csv(compute_dir+"ensemble_lstm_weights/scores.csv")
		station_scores = pretrained_scores[pretrained_scores['station_id']==station_code]

		best_members = station_scores.nlargest(5,'r2')
		print(best_members)


		df = pd.read_csv(compute_dir+"hindcast-catchment-data/{}{}.csv".format(station_code, f_dstring)).dropna()

		glofas_c = df['glofas_c'].values
		glofas_s = df['glofas_s'].values
		average = df['inflow_avg'].values

		era_vars = [ 'tp', 'sro', 'ssro', 'ro', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'u10',
				      'v10', 'd2m', 'sde', 'sf', 'smlt', 'snowc', 't2m', 'src', 'skt',
					  'stl1', 'slhf', 'e']

		predictors = np.array([glofas_c, glofas_s, average]+ [df[var].values for var in era_vars]).T

		obs_predictors = np.array([obs_glofas_c, obs_glofas_s, obs_average]+ [obs_df[var].values for var in era_vars]).T


		n_predictors = predictors.shape[-1]
		n_timestamp = 7*2


		sca = MinMaxScaler(feature_range = (0, 1))
		
		obs_predictors_scaled = sca.fit_transform(obs_predictors)
		predictors_scaled = sca.transform(predictors)
		
		scb = MinMaxScaler(feature_range = (0, 1))
		
		#predictand[np.isnan(predictand)] = np.mean(predictand[~np.isnan(predictand)])
		predictand_scaled = scb.fit_transform(predictand)
		
		

		testing_predictors = predictors_scaled.copy()
		

		X_test = data_split(testing_predictors, n_timestamp,)
		X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_predictors)

		json_file = open(compute_dir+'ensemble_lstm_weights/{}.json'.format(station_code), 'r')
		loaded_model_json = json_file.read()
		json_file.close()

		flows = []

		for member in best_members['ensemble_no'].values:
			model = model_from_json(loaded_model_json)
			model.load_weights(compute_dir+"ensemble_lstm_weights/{}_{:02d}.h5".format(station_code,member))

			y_predicted = model.predict(X_test)
					
			y_predicted_descaled = scb.inverse_transform(y_predicted)
			y_predicted_descaled = np.clip(y_predicted_descaled, a_min=0, a_max=None)
			flows.append(y_predicted_descaled.squeeze())

		ens_flows = np.array(flows).T
		ens_mean = np.mean(ens_flows,axis=-1)
		ens_dates = np.array([dt.strptime(d,"%Y-%m-%dT%H") for d in df.date.values])[-len(ens_mean):]
		
		
		#quick last minute bias correction using available obs if they exist
		gt = gt_df[gt_df.LocationID==station_code]
		if len(gt)==0:
			print("observations unavailable")
		else:
			gt_dates = np.array([dt.strptime(t, "%Y-%m-%dT%H") for t in gt.DateTime.values])

			itg = np.argmin(np.abs(gt_dates-start_date))
			td_min = start_date- gt_dates[itg]
			
			ite = np.argmin(np.abs(ens_dates-gt_dates[itg]))
			
			
			#print("gt_dates: ",gt_dates)
			#print("ens_dates: ",ens_dates)
			print("td_min: ",td_min)
			
			if np.abs(td_min)<=td(hours=24):
				gt_vals = gt.Value.values[itg-16:itg]
				if 'Ice' in gt_vals:
					delta = 0#ens_mean[it]
				else:
					en_val = np.mean(ens_mean[ite-8:ite])
					try: 
						gt_val = np.mean(gt_vals.astype(float))
						delta = en_val-gt_val
					except: delta =0
			
			print("delta = ",delta)
			if not np.isnan(delta):
				ens_mean -= delta
			else: print("observations unavailable")
		
		ens_mean = np.clip(ens_mean, a_min=0, a_max=None)
		
		
		for datum, date in zip(ens_mean, ens_dates):
			if date not in datelist: continue
			dstring = date.strftime("%Y-%m-%dT%H")
			f.write('{},{},{},{},{:5.2f},CFS\n'.format(dstring, station_code, fcast_dstring, handle, datum))
		


	f.close()






