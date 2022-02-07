import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("/home/users/rz908899/cluster/mr806421/us-rivers/compute/ensemble_lstm_weights/scores.csv")
print(df)
station_ids= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for sid,c in zip(station_ids,colors):
	df_sub = df[df['station_id']==sid]
	df_sub['rmse'] = np.sqrt(df_sub['mse'])/(3.28084**3)
	
	e10 = df_sub[(df_sub['ensemble_no']<50) | (df_sub['ensemble_no']>=55)]
	e50 = df_sub[(df_sub['ensemble_no']>=50) & (df_sub['ensemble_no']<55)]
	
	plt.scatter(np.log(e10['rmse']),e10['r2'], label=sid, c=c)
	plt.scatter(np.log(e50['rmse']),e50['r2'], c=c, edgecolors='k', linewidths=1,)# zorder=5)


plt.legend(loc='best', ncol=5)

plt.xlabel("ln(RMSE)")
plt.ylabel("Nash-Sutcliffe efficiency")
plt.gca().axhline(0, color='grey', linestyle=':')
plt.gca().axhline(1, color='grey', linestyle=':')

plt.show()
