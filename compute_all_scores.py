import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import glob
from datetime import datetime as dt, timedelta as td


def nash_sutcliffe(obs, sim, axis=None):
	qbar = np.mean(obs, axis=axis)
	num = np.sum((sim-obs)**2, axis=axis)
	dem = np.sum((obs-qbar)**2, axis=axis)
	return 1-(num/dem)

def kling_gupta(obs,sim, axis=None):
	r = pearsonr(sim, obs)[0]
	gamma = (r-1)**2
	alpha = ((sim.std()/obs.std())-1)**2
	beta = ((sim.mean()/obs.mean())-1)**2
	return 1-np.sqrt(alpha+beta+gamma)

compute_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/compute/"

station_ids= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']
exps = 'persistence','raw','bc','lstm'
modes = 'kge', 'nse'




for leadtime in np.arange(1,11):
	for exp in exps:
		scores = []
			
		flist = sorted(glob.glob("forecasts/{}_*.csv".format(exp)))[:-1]
		datelist = np.array([dt.strptime(f.split("/")[-1],"{}_%Y%m%dT%H.csv".format(exp)) for f in flist])
		
		start_date = datelist.min()
		end_date = datelist.max()+td(days=10)


		for station_id in station_ids:
	
			print(station_id)
			obs_df = pd.read_csv(compute_dir+'recent_obs.csv')
			obs = obs_df[obs_df.LocationID==station_id]
			obs_dates = np.array([dt.strptime(t,"%Y-%m-%dT%H") for t in obs.DateTime.values])
			obs_q = pd.to_numeric(obs.Value,errors='coerce').values
			
			obs_it = np.logical_and(obs_dates<=end_date, obs_dates>=start_date) 
			obs_dates = obs_dates[obs_it]
			obs_q = obs_q[obs_it]
			
			
			forecast_dates = []
			forecast_qs = []
			
			for f in flist:
				
				try: df_full = pd.read_csv(f)
				except: continue
				df = df_full[df_full.LocationID==station_id]
				
				
				it1 = 4*leadtime-3
				it2 = 4*leadtime+1
					
				df_lead = df.iloc[it1:it2]
				df_dates = np.array([dt.strptime(t,"%Y-%m-%dT%H") for t in df_lead.DateTime.values])
					
				forecast_dates.extend(df_dates)
				forecast_qs.extend(df_lead.Value.values)
					
			
			
			forecast_dates = np.array(forecast_dates)
			forecast_qs = np.array(forecast_qs)
			forecast_qs = np.nan_to_num(forecast_qs)

			no_nan = ~np.isnan(obs_q)
			it = np.in1d(obs_dates, forecast_dates)
			it2 = np.in1d(forecast_dates, obs_dates[no_nan])
			
			kge = kling_gupta(obs_q[np.logical_and(it,no_nan)], forecast_qs[it2])
			scores.append(kge)
			
			nse = nash_sutcliffe(obs_q[np.logical_and(it,no_nan)], forecast_qs[it2])
			scores.append(nse)
		
		score_strings = ["{:1.4f}".format(s) for s in scores]
		
		out_string = exp+"_{:02d},".format(leadtime)+",".join(score_strings)+"\n"
		print(out_string)
		f_scores = open('scores.csv', 'a+')
		f_scores.write(out_string)
		f_scores.close()







