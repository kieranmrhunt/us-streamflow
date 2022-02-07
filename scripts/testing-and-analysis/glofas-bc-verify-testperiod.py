import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D
from datetime import datetime as dt, timedelta as td
from netCDF4 import Dataset
from scipy.interpolate import interp1d

find = lambda x, arr: np.argmin(np.abs(x-arr))

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

s=3

gl_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/glofas/"
compute_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/compute/"

station_codes= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']
xlabels={'BNDN5':0.45, 'ARWN8':0.5, 'TCCC1':0.2, 'CARO2':0.45, 'ESSC2':0.45, 'NFDC1':0.2, 'LABW4':0.45, 'CLNK1':0.45, 'TRAC2':0.45, 'NFSW4':0.45}

df_locs = pd.read_csv("/home/users/rz908899/cluster/mr806421/us-rivers/compute/station_data.csv")

fig, axes = plt.subplots(5,2,sharex=True, figsize=(10,8))


for ax, station_code in list(zip(axes.ravel(),station_codes)):
	print(station_code)
	
	locx, locy = df_locs[df_locs['LocationID']==station_code][['Longitude', 'Latitude']].values.T
	locx = locx[0]
	locy = locy[0]


	df = pd.read_csv("/home/users/rz908899/cluster/mr806421/us-rivers/compute/catchment-data/{}.csv".format(station_code), parse_dates=[0,],
		             date_parser=lambda t:pd.to_datetime(str(t), format='%Y-%m-%dT%H'))[-365*8:]



	inflow = df['inflow_obs'].values
	average = df['inflow_avg'].values
	dates = df.date.values
	gl_flow = []
	outdates = []
	
	
	for date in dates:
		d = pd.to_datetime(date)
		hour = d.hour
		t = dt(d.year,d.month,d.day)
		t_glo = t+td(days=1)
		#print(station_code, d,)

		outdates.append(dt(d.year, d.month, d.day, d.hour))

		#collect glofas from following day, as it relays discharge from previous 24 hours
		gl_fname = gl_dir + t_glo.strftime("CEMS_ECMWF_dis24_%Y%m%d_glofas_v2.1.nc")
		gl_file = Dataset(gl_fname)

		lons = gl_file.variables['lon'][:] 
		lats = gl_file.variables['lat'][:]
		ix, iy = find(lons, locx), find(lats, locy)

		dis = gl_file.variables['dis24'][0, iy-s:iy+s, ix-s:ix+s]*(3.28084**3)

		quantiles = np.linspace(0,1,10000)
		obs_quantiles = np.load(compute_dir+"bias_matrices/{}_obs_quantile_{:02d}.npy".format(station_code,hour))
		gl_sorted, gl_quantile_sorted = np.load(compute_dir+"bias_matrices/{}_glofas_quantile_mapping.npy".format(station_code),)

		Tl, Yl, Xl = np.shape(gl_sorted)

		quantile_adjusted = np.zeros((Yl, Xl))

		for j in range(Yl):
			for i in range(Xl):
				gl_in = dis[j,i]
				
				uarr, uix = np.unique(gl_sorted[:,j,i], return_index=True)
				
				gl_quant = interp1d(gl_sorted[uix,j,i], gl_quantile_sorted[uix,j,i], fill_value='extrapolate')(gl_in)
				gl_out = interp1d(quantiles, obs_quantiles, fill_value='extrapolate')(gl_quant)
				quantile_adjusted[j,i] = gl_out



		kgo_popt = np.load(compute_dir+"bias_matrices/{}_kgo_popt.npy".format(station_code))

		reshaped_popt = np.reshape(kgo_popt, (Yl,Xl))
		kg_opt = np.sum(quantile_adjusted*reshaped_popt, axis=(-1,-2))
		kg_opt = np.clip(kg_opt, a_min=0, a_max=None)
		
		gl_flow.append(kg_opt)
	
	
	
	
	ens_mean = np.array(gl_flow)

	obs_flow = inflow.copy()

	valid = ~np.isnan(obs_flow)
	nse = r2_score(obs_flow[valid], ens_mean[valid])
	print("nse(r2)=" + str(nse))
	nse = nash_sutcliffe(obs_flow[valid], ens_mean[valid])
	print("nse=" + str(nse))
	kge = kling_gupta(obs_flow[valid], ens_mean[valid])
	print("kge=" + str(kge))

	


	ax.plot(dates,ens_mean/(3.28084**3), color='r')
	ax.plot(dates,obs_flow/(3.28084**3), color='k')

	x_label = xlabels[station_code]
	axtxt = '$\\mathbf{{{0}}}$\nNSE: {1:1.3f}\nKGE: {2:1.3f}'.format(station_code,nse,kge)	

	ax.text(x_label, 0.9, axtxt, transform=ax.transAxes, ha='left', va='top')
	ax.set_xlim([dates.min(),dates.max()])
	ax.tick_params(axis='x', rotation=50)

plt.setp(axes[:,0], ylabel='Flow (m$^3$ s$^{-1}$)')

legend_elements = [Line2D([0],[0], color='k',  label= 'Observed', ),
                   Line2D([0],[0], color='r',  label= 'GloFAS-ERA5 bias-corrected', ),
                   ]

fig.subplots_adjust(top=0.9, bottom=0.14)
fig.legend(handles=legend_elements, loc='lower center', frameon=True, ncol=2)


plt.show()








