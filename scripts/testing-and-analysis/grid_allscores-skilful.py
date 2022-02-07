import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta as td
from matplotlib.lines import Line2D
import glob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


station_ids= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']

scores = pd.read_csv("scores.csv")
exps = "persistence", "raw", "bc", "lstm"
colors = ['grey', u'#d62728', u'#ff7f0e', u'#1f77b4',]

fig, axes = plt.subplots(2,5, figsize=(12,6))
leadtimes = np.arange(1,11)

for nax, (ax, station_id) in enumerate(zip(axes.ravel(),station_ids)):
	for n,exp in enumerate(exps):
		df = scores[scores['model'].str.contains(exp)]
		#print(df)
		
		kges =  df['{}_kge'.format(station_id)].values
		nses =  df['{}_nse'.format(station_id)].values
	
		ax.plot(nses,kges,color=colors[n],lw=0.5)
		ax.scatter(nses,kges,c=colors[n],s=(11-leadtimes)*2)
	
	
	#ax.set_title(station_id)
	#ax.set_xscale('symlog',linthresh=1,linscale=5)
	#ax.set_yscale('symlog',linthresh=1,linscale=5)
	
	
	x1, x2 = ax.get_xlim()
	y1, y2 = ax.get_ylim()
	 
	
	ax.set_xticks([-1,-0.5,0,0.5,1])
	ax.set_xticklabels([-1,"",0,"",1])
		
	ax.set_yticks([-1,-0.5,0,0.25,0.5,0.75,1])
	ax.set_yticklabels(["","",0,"",0.5,"",1])
	
	
	
	ax.set_ylim(bottom=0,top=1)
	ax.set_xlim(left=-1,right=1)
	
	ax.grid(lw=0.3,c='grey')
	#ax.axhline(0,c='k',lw=0.5)
	ax.axvline(0,c='dimgrey',lw=0.5)
	#for n in [0.5, -0.5]:
	#	ax.axhline(n,c='grey',lw=0.5)
	#	ax.axvline(n,c='grey',lw=0.5)
	
	axtxt = station_id	
	t = ax.text(0.075, 0.95, axtxt, transform=ax.transAxes, ha='left', va='top')
	t.set_bbox(dict(facecolor='w',alpha=1))
	
	#ax.set_xlim([datelist.min(),datelist.max()+td(days=leadtimes[-1])])
	#ax.tick_params(axis='x', rotation=50)


fig.subplots_adjust()


plt.setp(axes[:,0], ylabel='KGE')
plt.setp(axes[-1,:], xlabel='NSE')

labels = 'Persistence', 'Raw GloFAS', 'Bias-corrected GloFAS', 'LSTM'
legend_elements = [Line2D([0],[0], color=colors[n],  label= labels[n]) for n in range(4)]
                   

fig.subplots_adjust(wspace=0.275, bottom=0.15)
fig.legend(handles=legend_elements, loc='lower center', frameon=True, ncol=4)


plt.show()

	
	
	












