import numpy as np
from datetime import datetime as dt, timedelta as td
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

gl_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/glofas/"
find = lambda x, arr: np.argmin(np.abs(x-arr))


station_id = 'TRAC2'


start_date = dt(2019,8,1,6)
ndays = 150

datelist = [start_date + i*td(hours=6) for i in range(4*ndays)]


df_locs = pd.read_csv("station_data.csv")
df_obs = pd.read_csv("flow.csv")
obs_dates = np.array([dt.strptime(t,'%Y-%m-%dT%H') for t in df_obs['DateTime'].values])

obs_flow_in = df_obs[station_id].values

locx, locy = df_locs[df_locs['LocationID']==station_id][['Longitude', 'Latitude']].values.T
locx = locx[0]
locy = locy[0]
	

obs_flow = []
raw_glofas_flow = []
bias_adjusted_glofas_flow = []

s=5

for date in datelist:
	print(date)
	hour = date.hour
	
	obs_flow.append(obs_flow_in[obs_dates==date])
	#print(obs_flow_in[obs_dates==date])
	
	gl_fname = gl_dir + date.strftime("CEMS_ECMWF_dis24_%Y%m%d_glofas_v2.1.nc")
	gl_file = Dataset(gl_fname)

	lons = gl_file.variables['lon'][:] 
	lats = gl_file.variables['lat'][:]
	ix, iy = find(lons, locx), find(lats, locy)
	dis = gl_file.variables['dis24'][0, iy-s:iy+s, ix-s:ix+s]*(3.28084**3)
	#print(np.shape(dis))
	raw_glofas_flow.append(dis[s,s])
	
	
	quantiles = np.linspace(0,1,1000)
	obs_quantiles = np.load("bias_matrices/{}_obs_quantile_{:02d}.npy".format(station_id,hour))
	gl_sorted, gl_quantile_sorted = np.load("bias_matrices/{}_glofas_quantile_mapping.npy".format(station_id),)
	
	
	#print(np.shape(obs_quantiles))
	#print(np.shape(gl_sorted))
	Tl, Yl, Xl = np.shape(gl_sorted)
	#print(np.shape(gl_quantile_sorted))
	
	quantile_adjusted = np.zeros((Yl, Xl))
	
	for j in range(Yl):
		for i in range(Xl):
			gl_in = dis[j,i]
			uarr, uix = np.unique(gl_sorted[:,j,i], return_index=True)
				
			gl_quant = interp1d(gl_sorted[uix,j,i], gl_quantile_sorted[uix,j,i], fill_value='extrapolate')(gl_in)
			gl_out = interp1d(quantiles, obs_quantiles, fill_value='extrapolate')(gl_quant)
			quantile_adjusted[j,i] = gl_out
	
	#bias_adjusted_glofas_flow.append(quantile_adjusted[s,s])
	
	
	kgo_popt = np.load("bias_matrices/{}_kgo_popt.npy".format(station_id))
	#print(np.shape(kgo_popt))
	
	reshaped_popt = np.reshape(kgo_popt, (Yl,Xl))
	kg_opt = np.nansum(quantile_adjusted*reshaped_popt, axis=(-1,-2))
	kg_opt = np.clip(kg_opt, a_min=0, a_max=None)
	
	bias_adjusted_glofas_flow.append(kg_opt)
	


plt.figure(figsize=(15,5))

plt.plot(datelist, obs_flow, 'k:', label='obs')
plt.plot(datelist, raw_glofas_flow, c='b', label='raw glofas')
plt.plot(datelist, bias_adjusted_glofas_flow, c='r', label='KG-opt glofas')
plt.legend(loc='best')
plt.show()

