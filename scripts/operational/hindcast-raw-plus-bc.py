import numpy as np
from datetime import datetime as dt, timedelta as td
from netCDF4 import Dataset, num2date
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter as gf
from matplotlib.colors import LinearSegmentedColormap

gl_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/daily-data/"
compute_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/compute/"
find = lambda x, arr: np.argmin(np.abs(x-arr))


#start_date = dt(dt.today().year, dt.today().month, dt.today().day)#-td(days=1)
#start_date = dt(2020,10,8)

for start_date in [dt(2020,9,1)+td(days=i) for i in range(395)]:
	ndays = 10 #length of forecast
	datelist = [start_date+td(hours=18) + i*td(hours=6) for i in range(0,4*ndays)]
	s=3 #bias matrix size - do not change

	handle_raw = 'TC+KHGMEC_raw'
	handle_bc = 'TC+KHGMEC_bc'

	df_locs = pd.read_csv(compute_dir+"station_data.csv")

	gl_fname = gl_dir + start_date.strftime("%Y%m%d_dis.nc")
	gl_file = Dataset(gl_fname)
	gl_tobj = gl_file.variables['time']
	gl_dates = num2date(gl_tobj[:], gl_tobj.units)
	print(gl_dates)

	f_raw = open(start_date.strftime('forecasts/raw_%Y%m%dT%H.csv'), 'w+')
	f_bc = open(start_date.strftime('forecasts/bc_%Y%m%dT%H.csv'), 'w+')
	f_raw.write('DateTime,LocationID,ForecastTime,VendorID,Value,Units\n')
	f_bc.write('DateTime,LocationID,ForecastTime,VendorID,Value,Units\n')


	fcast_dstring = start_date.strftime('%Y-%m-%dT%H')
	station_ids = ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']




	for station_id in station_ids:
		print(station_id)
		locx, locy = df_locs[df_locs['LocationID']==station_id][['Longitude', 'Latitude']].values.T
		locx = locx[0]
		locy = locy[0]
		
		spat_ix,spat_iy = df_locs[df_locs['LocationID']==station_id][['spat_ix', 'spat_iy']].values.T
		spat_ix, spat_iy = spat_ix[0], spat_iy[0]

		for date in datelist:
			hour = date.hour
			dstring = date.strftime('%Y-%m-%dT%H')

			lons = gl_file.variables['longitude'][:] 
			lats = gl_file.variables['latitude'][:]
			ix, iy = find(lons, locx), find(lats, locy)
			it_ = np.searchsorted(gl_dates, date)
			#print(ix, iy, spat_ix, spat_iy)
			# +1 below to reflect that values of dis for D are stored in D+1 in glofas
			it = np.clip(it_+1,0,len(gl_dates)-1)
			dis = gl_file.variables['dis24'][it, iy-s:iy+s, ix-s:ix+s]*(3.28084**3)

			raw_glofas_flow = gl_file.variables['dis24'][it, spat_iy, spat_ix]*(3.28084**3)
			#raw_glofas_flow = dis[s,s]

			quantiles = np.linspace(0,1,10000)
			obs_quantiles = np.load(compute_dir+"bias_matrices/{}_obs_quantile_{:02d}.npy".format(station_id,hour))
			gl_sorted, gl_quantile_sorted = np.load(compute_dir+"bias_matrices/{}_glofas_quantile_mapping.npy".format(station_id),)

			Tl, Yl, Xl = np.shape(gl_sorted)

			quantile_adjusted = np.zeros((Yl, Xl))

			for j in range(Yl):
				for i in range(Xl):
					gl_in = dis[j,i]
					
					uarr, uix = np.unique(gl_sorted[:,j,i], return_index=True)
					
					gl_quant = interp1d(gl_sorted[uix,j,i], gl_quantile_sorted[uix,j,i], fill_value='extrapolate')(gl_in)
					gl_out = interp1d(quantiles, obs_quantiles, fill_value='extrapolate')(gl_quant)
					quantile_adjusted[j,i] = gl_out


			kgo_popt = np.load(compute_dir+"bias_matrices/{}_kgo_popt.npy".format(station_id))

			reshaped_popt = np.reshape(kgo_popt, (Yl,Xl))
			kg_opt = np.sum(quantile_adjusted*reshaped_popt, axis=(-1,-2))
			kg_opt = np.clip(kg_opt, a_min=0, a_max=None)


			'''
			if station_id == 'LABW4':
				kg_opt = 0.75*raw_glofas_flow+0.25*kg_opt
			'''
			
			'''
			if station_id == 'NFDC1':
				kg_opt = 0.75*raw_glofas_flow+0.25*kg_opt
			'''
			
			
			
			bias_adjusted_glofas_flow = kg_opt
			print(date, it, iy, ix, raw_glofas_flow, bias_adjusted_glofas_flow)


			f_raw.write('{},{},{},{},{:5.2f},CFS\n'.format(dstring,station_id,fcast_dstring,handle_raw,raw_glofas_flow))
			f_bc.write('{},{},{},{},{:5.2f},CFS\n'.format(dstring,station_id,fcast_dstring,handle_bc,bias_adjusted_glofas_flow))




	f_raw.close()
	f_bc.close()

