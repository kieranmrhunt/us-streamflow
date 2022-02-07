import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import model_from_json

from numpy.random import seed
import tensorflow as tf
tf.compat.v1.random.set_random_seed(1)
seed(3)



station_ids= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']
station_code = 'BNDN5'


print(station_code)

df = pd.read_csv("catchment-data/{}.csv".format(station_code), parse_dates=[0,],
                 date_parser=lambda t:pd.to_datetime(str(t), format='%Y-%m-%dT%H'))

#df = df.dropna()



#inflow is the target (i.e. streamflow at a specific point.
#average is average for that station on that day over the last 10 years
# totaal is the total streamflow measured across all stations in the country

inflow = df['inflow_obs'].values
glofas_c = df['glofas_c'].values
glofas_s = df['glofas_s'].values
average = df['inflow_avg'].values


#era_vars = ['tp','sro','ssro','ro','swvl1','swvl2','swvl3','swvl4','u10']

era_vars = [ 'tp', 'sro', 'ssro', 'ro', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'u10',
              'v10', 'd2m', 'sde', 'sf', 'smlt', 'snowc', 't2m', 'src', 'skt',
			  'stl1', 'slhf', 'e']

#era_vars = ['u10',]


#inflow and total are rolled so that data from preceding day is used (i.e. not using x(t) to predict x(t))

#np.roll(inflow,1), np.roll(total,1)
predictors = np.array([glofas_c, glofas_s, average]+# np.roll(inflow,1), np.roll(total,1)]+
                      [df[var].values for var in era_vars]).T

print(predictors.shape)




predictand = np.array(inflow)[:,None]

n_predictors = predictors.shape[-1]
n_timestamp = 7*4
testing_days = 365*8

train_days = len(inflow)-testing_days



sca = MinMaxScaler(feature_range = (0, 1))
predictors_scaled = sca.fit_transform(predictors)

scb = MinMaxScaler(feature_range = (0, 1))
predictand_scaled = scb.fit_transform(predictand)

training_predictors = predictors_scaled[0:train_days]
training_predictand = predictand_scaled[0:train_days]
testing_predictors = predictors_scaled[train_days: train_days+testing_days]
testing_predictand = predictand_scaled[train_days: train_days+testing_days]



def data_split(predictors, predictand, n_timestamp, allow_nans=False):
	X = []
	y = []
	for i in range(len(predictors)):
		end_ix = i + n_timestamp
		if end_ix > len(predictors)-1:
			break
		# i to end_ix as input
		# end_ix as target output
		seq_x, seq_y = predictors[i:end_ix+1:2], predictand[end_ix]
		if not allow_nans:
			if np.isnan(seq_y): continue
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


X_train, y_train = data_split(training_predictors, training_predictand, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_predictors)

X_test, y_test = data_split(testing_predictors, testing_predictand, n_timestamp,)# allow_nans=True)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_predictors)



json_file = open('ensemble_lstm_weights/{}.json'.format(station_code), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("ensemble_lstm_weights/{}_45.h5".format(station_code))



y_predicted = model.predict(X_test)
y_predicted_descaled = scb.inverse_transform(y_predicted)
y_predicted_descaled = np.clip(y_predicted_descaled, a_min=0, a_max=None)

y_train_descaled = scb.inverse_transform(y_train)
y_test_descaled = scb.inverse_transform(y_test)
y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]
y_tested = y_test.ravel()


mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
r2 = r2_score(y_test_descaled, y_predicted_descaled)
print("mse=" + str(round(mse,2)))
print("r2=" + str(r2))






#np.save("temp_data/{}_vanilla".format(station_name), y_predicted_descaled)


plt.figure(figsize=(8,7))

plt.subplot(3, 1, 1)
plt.plot(inflow, color = 'black', linewidth=1, label = 'True value')
plt.ylabel("Inflow")
plt.xlabel("Day")
plt.title("All data")


plt.subplot(3, 2, 3)
plt.plot(y_test_descaled, color = 'black', linewidth=1, label = 'True value')
plt.plot(y_predicted_descaled, color = 'red',  linewidth=1, label = 'Predicted')
plt.legend(frameon=False)
plt.ylabel("Inflow")
plt.xlabel("Day")
plt.title("Predicted data ({} timesteps)".format(testing_days))

#np.save("temp/{}.npy".format(station), y_predicted_descaled) 

plt.subplot(3, 2, 4)
plt.plot(y_test_descaled[:75], color = 'black', linewidth=1, label = 'True value')
plt.plot(y_predicted_descaled[:75], color = 'red', label = 'Predicted')
plt.legend(frameon=False)
plt.ylabel("Inflow")
plt.xlabel("Day")
plt.title("Predicted data (first 75 timesteps)")


plt.subplot(3, 3, 8)
plt.plot(y_test_descaled-y_predicted_descaled, color='black')
plt.ylabel("Residual")
plt.xlabel("Day")
plt.title("Residual plot")

plt.subplot(3, 3, 9)
plt.scatter(y_predicted_descaled, y_test_descaled, s=2, color='black')
plt.ylabel("Y true")
plt.xlabel("Y predicted")
plt.title("Scatter plot")

plt.subplots_adjust(hspace = 0.5, wspace=0.3)




mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
r2 = r2_score(y_test_descaled, y_predicted_descaled)
print("mse=" + str(round(mse,2)))
print("r2=" + str(r2))

plt.show()













