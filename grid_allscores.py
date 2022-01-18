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
colors = ['k',u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

fig, axes = plt.subplots(2,5, figsize=(12,6))
leadtimes = np.arange(1,11)

for ax, station_id in list(zip(axes.ravel(),station_ids)):
	for n,exp in enumerate(exps):
		df = scores[scores['model'].str.contains(exp)]
		#print(df)
		
		kges =  df['{}_kge'.format(station_id)].values
		nses =  df['{}_nse'.format(station_id)].values
	
		ax.plot(nses,kges,color=colors[n],lw=0.5)
		ax.scatter(nses,kges,c=colors[n],s=(11-leadtimes)*2)
	
	
	#ax.set_title(station_id)
	ax.set_xscale('symlog',linthresh=1,linscale=5)
	ax.set_yscale('symlog',linthresh=1,linscale=5)
	
	
	x1, x2 = ax.get_xlim()
	y1, y2 = ax.get_ylim()
	 
	
	ax.set_yticks([-1e7,-1e6,-1e5,-1e4,-1e3,-1e2,-1e1,-1,-0.5,0,0.5,1])
	ax.set_yticklabels(["","","","","","","",-1,"",0,0.5,1])
		
	ax.set_xticks([-1e7,-1e6,-1e5,-1e4,-1e3,-1e2,-1e1,-1,-0.5,0,0.5,1])
	ax.set_xticklabels(["","","","","","","",-1,"",0,"",1])
	
	if station_id in ['LABW4','TRAC2','ESSC2']:
		ax.set_xticklabels(["","","","","","","",-1,"",0,0.5,1])
	
	if station_id == 'NFSW4':
		ax.set_yticks([-0.5,0,0.5,0.75,0.9,1])
		ax.set_yticklabels(["",0,0.5,0.75,0.9,1])
		
		ax.set_xticks([-0.5,0,0.5,0.75,1])
		ax.set_xticklabels(["",0,0.5,0.75,1])
	
	
	ax.set_ylim(bottom=y1,top=1)
	ax.set_xlim(left=x1,right=1)
	
	ax.grid(lw=0.25,c='lightgrey')
	ax.axhline(0,c='k',lw=0.5)
	ax.axvline(0,c='k',lw=0.5)
	for n in [0.5, -0.5]:
		ax.axhline(n,c='grey',lw=0.5)
		ax.axvline(n,c='grey',lw=0.5)
	
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

	
	
	












