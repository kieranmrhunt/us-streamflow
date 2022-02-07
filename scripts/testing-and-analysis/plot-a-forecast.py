import pandas as pd
from datetime import datetime as dt, timedelta as td
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

#start_date = dt(dt.today().year, dt.today().month, dt.today().day)-td(days=10)
start_date = dt(2021,5,1)
raw_df = pd.read_csv(start_date.strftime('forecasts/raw_%Y%m%dT%H.csv'))
bc_df = pd.read_csv(start_date.strftime('forecasts/bc_%Y%m%dT%H.csv'))
lstm_df = pd.read_csv(start_date.strftime('forecasts/lstm_%Y%m%dT%H.csv'))
pers_df = pd.read_csv(start_date.strftime('forecasts/persistence_%Y%m%dT%H.csv'))

obs_df = pd.read_csv('../compute/recent_obs.csv')

station_codes= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']

fig, axes = plt.subplots(2, 5, sharex=True, figsize=(12,6))


for station_code, ax in zip(station_codes, axes.ravel()):

	raw = raw_df[raw_df.LocationID==station_code]
	bc = bc_df[bc_df.LocationID==station_code]
	lstm = lstm_df[lstm_df.LocationID==station_code]
	pers = pers_df[lstm_df.LocationID==station_code]
	
	dates = [dt.strptime(t, "%Y-%m-%dT%H") for t in raw.DateTime.values]

	obs = obs_df[obs_df.LocationID==station_code]
	obs_dates = [dt.strptime(t, "%Y-%m-%dT%H") for t in obs.DateTime.values]
	
	
	ax.plot(dates, pers.Value/(3.28084**3), color='grey',)
	ax.plot(dates, raw.Value/(3.28084**3), color=u'#d62728')
	ax.plot(dates, bc.Value/(3.28084**3), color=u'#ff7f0e')
	ax.plot(dates, lstm.Value/(3.28084**3), color=u'#1f77b4')
	
	
	for obs_date, obs_value in zip(obs_dates, obs.Value.values):
		if obs_date not in dates: continue
		if obs_value =='Ice' : ax.plot(obs_date, 0, 'bx')
		elif obs_value in ['Eqp',]: continue
		else: ax.plot(obs_date, float(obs_value)/(3.28084**3), 'kx')
		
	#ax.plot(, 'k+')
	
	if station_code == "ARWN8":
		ax.set_yticks([0,0.04,0.08, 0.12])
	if station_code == "CLNK1":
		ax.set_yticks([0.1, 0.2, 0.3])
	
	ax.set_title(station_code)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
	plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)

	ax.set_xlim([np.min(dates), np.max(dates)])

for ax in axes[:,0]:
	ax.set_ylabel('Streamflow (m$^3$ s$^{-1}$)')


fig.subplots_adjust(bottom=0.17, hspace=0.195, wspace=0.285)


ax=axes[1,2]
handles = [Line2D([0], [0], color='grey', label='Persistence'),
	       Line2D([0], [0], color='#d62728', label='Raw GloFAS'),
           Line2D([0], [0], color='#ff7f0e', label='Bias-corrected GloFAS'),
           Line2D([0], [0], color='#1f77b4', label='LSTM'),
           Line2D([0], [0], color='k', marker='x', lw=0, label='USGS observation'),
]
ax.legend(handles = handles,loc='upper center', ncol=5,  bbox_to_anchor=(0.5, -0.275))

plt.suptitle('Forecast issued: {}'.format(start_date+td(hours=12)))
plt.show()
	


























