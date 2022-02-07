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

start_date = dt(dt.today().year, dt.today().month, dt.today().day)#-td(days=1)
#start_date = dt(2021,6,9)


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

f_raw = open(start_date.strftime(compute_dir+'operational_output/raw_%Y%m%dT%H.csv'), 'w+')
f_bc = open(start_date.strftime(compute_dir+'operational_output/bc_%Y%m%dT%H.csv'), 'w+')
f_raw.write('DateTime,LocationID,ForecastTime,VendorID,Value,Units\n')
f_bc.write('DateTime,LocationID,ForecastTime,VendorID,Value,Units\n')


fcast_dstring = start_date.strftime('%Y-%m-%dT%H')
station_ids = ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']




for station_id in station_ids:
	#print(station_id)
	locx, locy = df_locs[df_locs['LocationID']==station_id][['Longitude', 'Latitude']].values.T
	locx = locx[0]
	locy = locy[0]
	#print(locx, locy)
	spat_ix,spat_iy = df_locs[df_locs['LocationID']==station_id][['spat_ix', 'spat_iy']].values.T
	spat_ix, spat_iy = spat_ix[0], spat_iy[0]
	#print(spat_ix, spat_iy)
	historic = pd.read_csv(compute_dir + "catchment-data/{}.csv".format(station_id)).dropna()
	historic_obs = historic['inflow_obs'].values
	historic_glf = historic['glofas_s'].values

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
		
		#print(lons[spat_ix], lats[spat_iy])	
		#print(gl_file.variables['dis24'][:, spat_iy, spat_ix]*(3.28084**3))
		q = np.mean(historic_glf<=raw_glofas_flow)
		Q = np.nanquantile(historic_obs, q)
		bias_adjusted_glofas_flow = np.clip(Q, a_min=0, a_max=None)
		
		#bias_adjusted_glofas_flow = np.array(bias_adjusted_flow)
		#print(date, it, iy, ix, raw_glofas_flow, bias_adjusted_glofas_flow)


		f_raw.write('{},{},{},{},{:5.2f},CFS\n'.format(dstring,station_id,fcast_dstring,handle_raw,raw_glofas_flow))
		f_bc.write('{},{},{},{},{:5.2f},CFS\n'.format(dstring,station_id,fcast_dstring,handle_bc,bias_adjusted_glofas_flow))




f_raw.close()
f_bc.close()

