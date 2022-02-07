import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D

def data_split(predictors,predictand, n_timestamp):
	X = []
	y = []
	for i in range(len(predictors)):
		end_ix = i + n_timestamp
		if end_ix > len(predictors)-1:
			break
		seq_x, seq_y = predictors[i:end_ix+1:2], predictand[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

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





station_codes= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']
xlabels={'BNDN5':0.45, 'ARWN8':0.5, 'TCCC1':0.2, 'CARO2':0.45, 'ESSC2':0.45, 'NFDC1':0.2, 'LABW4':0.45, 'CLNK1':0.45, 'TRAC2':0.45, 'NFSW4':0.45}

fig, axes = plt.subplots(5,2,sharex=True, figsize=(10,8))


for ax, station_code in list(zip(axes.ravel(),station_codes)):


	pretrained_scores = pd.read_csv("/home/users/rz908899/cluster/mr806421/us-rivers/compute/ensemble_lstm_weights/scores.csv")
	station_scores = pretrained_scores[pretrained_scores['station_id']==station_code]

	best_members = station_scores.nlargest(5,'r2')#['ensemble_no'].values
	print(best_members)


	df = pd.read_csv("/home/users/rz908899/cluster/mr806421/us-rivers/compute/catchment-data/{}.csv".format(station_code), parse_dates=[0,],
		             date_parser=lambda t:pd.to_datetime(str(t), format='%Y-%m-%dT%H'))

	inflow = df['inflow_obs'].values
	glofas_c = df['glofas_c'].values
	glofas_s = df['glofas_s'].values
	average = df['inflow_avg'].values

	era_vars = [ 'tp', 'sro', 'ssro', 'ro', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'u10',
		          'v10', 'd2m', 'sde', 'sf', 'smlt', 'snowc', 't2m', 'src', 'skt',
				  'stl1', 'slhf', 'e']

	predictors = np.array([glofas_c, glofas_s, average]+ [df[var].values for var in era_vars]).T
	predictand = np.array(inflow)[:,None]

	n_predictors = predictors.shape[-1]
	n_timestamp = 7*4
	testing_days = 365*8
	#testing_days = 40

	train_days = len(inflow)-testing_days

	sca = MinMaxScaler(feature_range = (0, 1))
	predictors_scaled = sca.fit_transform(predictors)

	scb = MinMaxScaler(feature_range = (0, 1))
	predictand_scaled = scb.fit_transform(predictand)


	testing_predictors = predictors_scaled[train_days: train_days+testing_days]
	testing_predictand = predictand_scaled[train_days: train_days+testing_days]

	X_test, y_test = data_split(testing_predictors, testing_predictand, n_timestamp,)
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_predictors)

	json_file = open("/home/users/rz908899/cluster/mr806421/us-rivers/compute/ensemble_lstm_weights/{}.json".format(station_code), 'r')
	loaded_model_json = json_file.read()
	json_file.close()

	flows = []

	for member in best_members['ensemble_no'].values:
		model = model_from_json(loaded_model_json)
		model.load_weights("/home/users/rz908899/cluster/mr806421/us-rivers/compute/ensemble_lstm_weights/{}_{:02d}.h5".format(station_code,member))

		y_predicted = model.predict(X_test)
		y_predicted_descaled = scb.inverse_transform(y_predicted)
		y_predicted_descaled = np.clip(y_predicted_descaled, a_min=0, a_max=None)

		flows.append(y_predicted_descaled.squeeze()[::2])

	ens_flows = np.array(flows).T
	ens_mean = np.mean(ens_flows,axis=-1)

	obs_flow = scb.inverse_transform(y_test).squeeze()[::2]

	valid = ~np.isnan(obs_flow)
	nse = r2_score(obs_flow[valid], ens_mean[valid])
	print("nse(r2)=" + str(nse))
	nse = nash_sutcliffe(obs_flow[valid], ens_mean[valid])
	print("nse=" + str(nse))
	kge = kling_gupta(obs_flow[valid], ens_mean[valid])
	print("kge=" + str(kge))

	dates = df.date.values[::2][-len(ens_mean):]

	ax.plot(dates,ens_flows/(3.28084**3), color='grey', lw=.5)
	ax.plot(dates,ens_mean/(3.28084**3), color='r')
	ax.plot(dates,obs_flow/(3.28084**3), color='k')

	x_label = xlabels[station_code]
	axtxt = '$\\mathbf{{{0}}}$\nNSE: {1:1.3f}\nKGE: {2:1.3f}'.format(station_code,nse,kge)	

	ax.text(x_label, 0.9, axtxt, transform=ax.transAxes, ha='left', va='top')
	ax.set_xlim([dates.min(),dates.max()])
	ax.tick_params(axis='x', rotation=50)

plt.setp(axes[:,0], ylabel='Flow (m$^3$ s$^{-1}$)')

legend_elements = [Line2D([0],[0], color='k',  label= 'Observed', ),
                   Line2D([0],[0], color='r',  label= 'LSTM ensemble mean', ),
                   Line2D([0],[0], color='grey', lw=0.5,  label= 'LSTM ensemble member'),]

fig.subplots_adjust(top=0.9, bottom=0.14)
fig.legend(handles=legend_elements, loc='lower center', frameon=True, ncol=3)


plt.show()








