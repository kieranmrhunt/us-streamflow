from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta as td
import os
import sys

def fill_nans(y):
	filled = y.astype(float)
	nans, x = np.isnan(filled), lambda z: z.nonzero()[0]
	filled[nans]= np.interp(x(nans), x(~nans), filled[~nans])
	return filled

compute_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/compute/"
data_dir = '/home/users/rz908899/cluster/mr806421/us-rivers/daily-data/'

idate = sys.argv[1]
#start_date = dt(dt.today().year, dt.today().month, dt.today().day)
start_date = dt.strptime(idate, "%Y%m%d")

era_vars = [ 'total_precipitation', 'surface_runoff', 'sub_surface_runoff', 'runoff',
                  'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
            	  '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            	  'snow_depth', 'snowfall', 'snowmelt',
				  '2m_temperature','skin_reservoir_content', 
				  'skin_temperature','soil_temperature_level_1',
            	  'surface_latent_heat_flux', 'total_evaporation', 'snow_cover'
        	]

era_codes = [ 'tp', 'sro', 'ssro', 'ro', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'u10',
              'v10', 'd2m', 'sde', 'sf', 'smlt', 't2m', 'src', 'skt',
			  'stl1', 'slhf', 'e', 'snowc']

accumulated_codes = ['tp','sro','ssro','ro','smlt','slhf','e', 'sf']


station_data = pd.read_csv(compute_dir+'station_data.csv')
obs = pd.read_csv(compute_dir+"flow.csv",)
obs_dates = np.array([dt.strptime(str(t), '%Y-%m-%dT%H') for t in obs['DateTime'].values])

posterior_dates = [start_date-td(hours=7*24-i) for i in range(0,8*24,6)]
posterior_days = np.unique([dt(d.year, d.month, d.day) for d in posterior_dates])

anterior_dates = [start_date+td(hours=i) for i in range(24,11*24,6)]
anterior_days = np.unique([dt(d.year, d.month, d.day) for d in anterior_dates])

output_dates = np.array(posterior_dates + anterior_dates)
output_days = np.unique([dt(d.year, d.month, d.day) for d in output_dates])


dummy_axis = np.linspace(0,1,len(output_dates))



for index, station in station_data.iterrows():
	
	
	parsed_data = {var:np.ones_like(output_dates)*np.nan 
                   for var in era_codes+['glofas_c','glofas_s','inflow_avg']}
	
	station_code = station['LocationID']
	
	print(station_code)
	
	blon, blat = station['Longitude'], station['Latitude']
	basin_mask = np.load(compute_dir+"catchment-masks/{}.npy".format(station_code))
	#print(np.sum(basin_mask))
	
	gix, giy = station['centr_ix'], station['centr_iy']
	gixs, giys = station['spat_ix'], station['spat_iy']

	obs_flows = obs[station_code].values

	
	#process regular variables
	for var, code in zip(era_vars, era_codes):		
		if code=='snowc': continue
	
		print(station_code,var,code)
		for d in output_days:
			if d in posterior_days: 
				fname = data_dir+d.strftime('%Y%m%d_{}_day1.nc'.format(var))
				fname_prev = data_dir+(d-td(days=1)).strftime('%Y%m%d_{}_day2.nc'.format(var))
				print(fname)
				#print(fname_prev)
			
			elif d in anterior_days:
				lead = 1+(d-start_date).days
				fname = data_dir+start_date.strftime('%Y%m%d_{}_day{}.nc'.format(var,lead))
				fname_prev = data_dir+start_date.strftime('%Y%m%d_{}_day{}.nc'.format(var,lead-1))
				print(fname)
				#print(fname_prev)
			
			if not os.path.isfile(fname): print("could not locate {}".format(fname)); continue
			
				
			try: infile = Dataset(fname)
			except: continue
			
			_code = 'sd' if code=='sde' else code
			try: data = infile.variables[_code][:]
			except: continue
			
			if code in accumulated_codes:
				#data = np.diff(data,axis=0)
				try:
					infile_prev = Dataset(fname_prev)
					data_prev = infile_prev.variables[_code][0]
				except: continue
				
				if d in posterior_days:
					#print("post", np.mean(data[0]), np.mean(data_prev))
					data[1:] = data[1:]-data[0]
					data[0] = data_prev
					
				if d in anterior_days:
					#print("ant", np.mean(data[0]), np.mean(data_prev))
					data[1:] = data[1:]-data[0]
					data[0] = data[0]-data_prev
			
			data_basin = np.sum(data*basin_mask[None, :, :], axis=(-1,-2))/np.sum(basin_mask)
			ftimeobj = infile.variables['time']
			ftimes = num2date(ftimeobj[:], ftimeobj.units)
			
			for ftime, datum in zip(ftimes,data_basin):
				it = np.argmin(np.abs(ftime - output_dates))
				#print(it)
				parsed_data[code][it] = datum
		
			#process snow cover by approximating from bool of snow depth
			if code=='sde':
				data = infile.variables['sd'][:].astype(bool).astype(float)*100
				data_basin = np.sum(data*basin_mask[None, :, :], axis=(-1,-2))/np.sum(basin_mask)
				for ftime, datum in zip(ftimes,data_basin):
					it = np.argmin(np.abs(ftime - output_dates))
					parsed_data['snowc'][it] = datum
		
		
		parsed_data[code] = fill_nans(parsed_data[code])
	parsed_data['snowc'] = fill_nans(parsed_data['snowc'])


	#process glofas
	#archived forecast
	for day in posterior_days:
		fname = data_dir+day.strftime('%Y%m%d_dis.nc')
		gl_file = Dataset(fname)
		dis = gl_file.variables['dis24'][0, giy, gix]*(3.28084**3)
		dis_s = gl_file.variables['dis24'][0, giys, gixs]*(3.28084**3)
		
		gl_time = gl_file.variables['time']
		gl_times = num2date(gl_time[:], gl_time.units)[0]
		it = np.argmin(np.abs(gl_times - output_dates))
		
		parsed_data['glofas_c'][it] = dis
		parsed_data['glofas_s'][it] = dis_s
	#operational forecast
	fname = data_dir+start_date.strftime('%Y%m%d_dis.nc')
	gl_file = Dataset(fname)
	dis = gl_file.variables['dis24'][:, giy, gix]*(3.28084**3)
	dis_s = gl_file.variables['dis24'][:, giys, gixs]*(3.28084**3)
	gl_time = gl_file.variables['time']
	gl_times = num2date(gl_time[:], gl_time.units)
	for ftime, datum, datum_s in zip(gl_times,dis,dis_s):
		it = np.argmin(np.abs(ftime - output_dates))
		parsed_data['glofas_c'][it] = datum
		parsed_data['glofas_s'][it] = datum_s
	
	
	#print(parsed_data['glofas_c'])
	parsed_data['glofas_s'] = fill_nans(parsed_data['glofas_s'])
	parsed_data['glofas_c'] = fill_nans(parsed_data['glofas_c'])
	#print(parsed_data['glofas_c'])
	
	
	#extract average inflow for day of year
	catchment_df = pd.read_csv(compute_dir+'catchment-data/{}.csv'.format(station_code))
	date_strings = catchment_df.date.values
	averages = catchment_df.inflow_avg.values
	
	for n,d in enumerate(output_dates):
		dstring = d.strftime("2016-%m-%dT%H")
		it = date_strings==dstring
		avg = averages[it]
		parsed_data['inflow_avg'][n] = avg.squeeze()
	
	dstring = start_date.strftime('%Y%m%dT%H')
	
	
	df = pd.DataFrame()
	print(len(output_dates))
	df['date'] = output_dates

	for var in ['glofas_c','glofas_s','inflow_avg']:
		df[var] = parsed_data[var]

	for code in era_codes:
		df[code] = parsed_data[code]

	print(df)
	
	
	
	df.to_csv(compute_dir+"hindcast-catchment-data/{}{}.csv".format(station_code,dstring), date_format='%Y-%m-%dT%H', index=False)
	
		












