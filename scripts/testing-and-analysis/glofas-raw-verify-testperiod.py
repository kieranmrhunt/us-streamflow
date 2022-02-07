import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D


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
	print(station_code)



	df = pd.read_csv("/home/users/rz908899/cluster/mr806421/us-rivers/compute/catchment-data/{}.csv".format(station_code), parse_dates=[0,],
		             date_parser=lambda t:pd.to_datetime(str(t), format='%Y-%m-%dT%H'))[-365*8:]

	inflow = df['inflow_obs'].values/(3.28084**3)
	average = df['inflow_avg'].values/(3.28084**3)
	glofas = df['glofas_s'].values/(3.28084**3)
	
	
	ens_mean = glofas.copy()
	obs_flow = inflow.copy()

	valid = ~np.isnan(obs_flow)
	nse = r2_score(obs_flow[valid], ens_mean[valid])
	print("nse(r2)=" + str(nse))
	nse = nash_sutcliffe(obs_flow[valid], ens_mean[valid])
	print("nse=" + str(nse))
	kge = kling_gupta(obs_flow[valid], ens_mean[valid])
	print("kge=" + str(kge))

	dates = df.date.values


	ax.plot(dates,ens_mean, color='r')
	ax.plot(dates,obs_flow, color='k')

	x_label = xlabels[station_code]
	axtxt = '$\\mathbf{{{0}}}$\nNSE: {1:1.3f}\nKGE: {2:1.3f}'.format(station_code,nse,kge)	

	ax.text(x_label, 0.9, axtxt, transform=ax.transAxes, ha='left', va='top')
	ax.set_xlim([dates.min(),dates.max()])
	ax.tick_params(axis='x', rotation=50)

plt.setp(axes[:,0], ylabel='Flow (m$^3$ s$^{-1}$)')

legend_elements = [Line2D([0],[0], color='k',  label= 'Observed', ),
                   Line2D([0],[0], color='r',  label= 'GloFAS-ERA5 raw', ),
                   ]

fig.subplots_adjust(top=0.9, bottom=0.14)
fig.legend(handles=legend_elements, loc='lower center', frameon=True, ncol=2)


plt.show()








