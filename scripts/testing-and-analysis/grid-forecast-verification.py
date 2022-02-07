import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta as td
from matplotlib.lines import Line2D
import glob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def kling_gupta(obs,sim, axis=None):
	r = pearsonr(sim, obs)[0]
	gamma = (r-1)**2
	alpha = ((sim.std()/obs.std())-1)**2
	beta = ((sim.mean()/obs.mean())-1)**2
	return 1-np.sqrt(alpha+beta+gamma)



station_ids= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']

compute_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/compute/"

exp='lstm' #raw,bc,lstm

flist = sorted(glob.glob("forecasts/{}_*.csv".format(exp)))[:-1]
datelist = np.array([dt.strptime(f.split("/")[-1],"{}_%Y%m%dT%H.csv".format(exp)) for f in flist])

xlabels={'BNDN5':0.02, 'ARWN8':0.02, 'TCCC1':0.65, 'CARO2':0.02, 'ESSC2':0.02, 'NFDC1':0.65, 'LABW4':0.02, 'CLNK1':0.02, 'TRAC2':0.02, 'NFSW4':0.02}


leadtimes = 2,5,8
colors = '#1f77b4', '#ff7f0e', '#bcbd22'

start_date = datelist.min()
end_date = datelist.max()+td(days=leadtimes[-1])

d = start_date
obs_dates_full = []
while d<=end_date:
	obs_dates_full.append(d)
	d+=td(hours=6)
obs_dates_full = np.array(obs_dates_full)
print(obs_dates_full)

fig, axes = plt.subplots(5,2,sharex=True, figsize=(10,8))

for ax, station_id in list(zip(axes.ravel(),station_ids)):
	
	print(station_id)
	obs_df = pd.read_csv(compute_dir+'recent_obs.csv')
	obs = obs_df[obs_df.LocationID==station_id]
	obs_dates = np.array([dt.strptime(t,"%Y-%m-%dT%H") for t in obs.DateTime.values])
	obs_q = pd.to_numeric(obs.Value,errors='coerce').values
	
	obs_it = np.logical_and(obs_dates<=end_date, obs_dates>=start_date) 
	obs_dates = obs_dates[obs_it]
	obs_q = obs_q[obs_it]
	
	obs_q_full = []
	
	for date in obs_dates_full:
		if date in obs_dates:
			it = np.searchsorted(obs_dates, date)
			obs_q_full.append(obs_q[it])
		else:
			obs_q_full.append(np.nan)

	obs_q_full = np.array(obs_q_full)
	
	
	forecast_dates = [[], [], []]
	forecast_qs = [[], [], []]
	
	for f in flist:
		
		try: df_full = pd.read_csv(f)
		except: continue
		df = df_full[df_full.LocationID==station_id]
		
		for nL, L in enumerate(leadtimes):
			it1 = 4*L-3
			it2 = 4*L+1
			
			df_lead = df.iloc[it1:it2]
			df_dates = np.array([dt.strptime(t,"%Y-%m-%dT%H") for t in df_lead.DateTime.values])
			
			forecast_dates[nL].extend(df_dates)
			forecast_qs[nL].extend(df_lead.Value.values)
			
	
	
	forecast_dates = np.array(forecast_dates)
	forecast_qs = np.array(forecast_qs)
	forecast_qs = np.nan_to_num(forecast_qs)

	ax.plot(obs_dates_full, obs_q_full/(3.28084**3), 'k-', lw=1.5)
	
	kges = []
	
	
	for n in range(len(leadtimes)):
		ax.plot(forecast_dates[n], forecast_qs[n]/(3.28084**3), color = colors[n],lw=1, zorder=10-n)
		
		no_nan = ~np.isnan(obs_q)
		it = np.in1d(obs_dates, forecast_dates[n])
		it2 = np.in1d(forecast_dates[n], obs_dates[no_nan])
		
		
		kge = kling_gupta(obs_q[np.logical_and(it,no_nan)], forecast_qs[n][it2])
		kges.append(kge)
	
	print(kges)
	
	x_label = xlabels[station_id]
	diff = 0.15
	
	axtxt = '$\\mathbf{{{0}}}$'.format(station_id)	
	ax.text(x_label, 0.95, axtxt, transform=ax.transAxes, ha='left', va='top')
	
	axtxt = '{0}-day KGE: {1:1.3f}'.format(leadtimes[0],kges[0])	
	ax.text(x_label, 0.95-diff, axtxt, transform=ax.transAxes, ha='left', va='top', color=colors[0])
	
	axtxt = '{0}-day KGE: {1:1.3f}'.format(leadtimes[1],kges[1])	
	ax.text(x_label, 0.95-2*diff, axtxt, transform=ax.transAxes, ha='left', va='top', color=colors[1])
	
	axtxt = '{0}-day KGE: {1:1.3f}'.format(leadtimes[2],kges[2])	
	ax.text(x_label, 0.95-3*diff, axtxt, transform=ax.transAxes, ha='left', va='top', color=colors[2])
	
	
	ax.set_xlim([datelist.min(),datelist.max()+td(days=leadtimes[-1])])
	ax.tick_params(axis='x', rotation=50)



plt.setp(axes[:,0], ylabel='Flow (m$^3$ s$^{-1}$)')

legend_elements = [Line2D([0],[0], color='k',  label= 'Observed', )] + [Line2D([0],[0], color=colors[n],  label= '{}-day LSTM forecast'.format(leadtimes[n])) for n in range(3)]
                   

fig.subplots_adjust(top=0.9, bottom=0.18)
fig.legend(handles=legend_elements, loc='lower center', frameon=True, ncol=2)


plt.show()

	
	
	












