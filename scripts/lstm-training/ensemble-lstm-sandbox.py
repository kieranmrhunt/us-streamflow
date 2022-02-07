import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler

def data_split(predictors,predictand, n_timestamp):
	X = []
	y = []
	for i in range(len(predictors)):
		end_ix = i + n_timestamp
		if end_ix > len(predictors)-1:
			break
		seq_x, seq_y = predictors[i:end_ix+1], predictand[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)




station_codes= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']


station_code = 'ESSC2'

pretrained_scores = pd.read_csv("ensemble_lstm_weights/scores.csv")
station_scores = pretrained_scores[pretrained_scores['station_id']==station_code]

best_members = station_scores.nlargest(5,'r2')#['ensemble_no'].values
print(best_members)


df_obs = pd.read_csv("catchment-data/{}.csv".format(station_code), parse_dates=[0,],
                 date_parser=lambda t:pd.to_datetime(str(t), format='%Y-%m-%dT%H'))

df = pd.read_csv("operational-catchment-data/{}20191021T00.csv".format(station_code), 					 )



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
n_timestamp = 7*2
#testing_days = 365*8
testing_days = 40

train_days = len(average)-testing_days

sca = MinMaxScaler(feature_range = (0, 1))
predictors_scaled = sca.fit_transform(predictors)

scb = MinMaxScaler(feature_range = (0, 1))
predictand_scaled = scb.fit_transform(predictand)


testing_predictors = predictors_scaled[train_days: train_days+testing_days]
testing_predictand = predictand_scaled[train_days: train_days+testing_days]

X_test, y_test = data_split(testing_predictors, testing_predictand, n_timestamp,)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_predictors)

json_file = open('ensemble_lstm_weights/{}.json'.format(station_code), 'r')
loaded_model_json = json_file.read()
json_file.close()

flows = []

for member in best_members['ensemble_no'].values:
	model = model_from_json(loaded_model_json)
	model.load_weights("ensemble_lstm_weights/{}_{:02d}.h5".format(station_code,member))

	y_predicted = model.predict(X_test)
	y_predicted_descaled = scb.inverse_transform(y_predicted)
	y_predicted_descaled = np.clip(y_predicted_descaled, a_min=0, a_max=None)

	flows.append(y_predicted_descaled.squeeze()[::])

ens_flows = np.array(flows).T
ens_mean = np.mean(ens_flows,axis=-1)

'''
obs_flow = scb.inverse_transform(y_test).squeeze()[::]

valid = ~np.isnan(obs_flow)
nse = r2_score(obs_flow[valid], ens_mean[valid])
print("nse=" + str(nse))
'''
dates = df.date.values[::][-len(ens_mean):]

plt.plot(dates,ens_flows, color='grey', lw=.5)
plt.plot(dates,ens_mean, color='r')
#plt.plot(dates,obs_flow, color='k')



plt.show()








