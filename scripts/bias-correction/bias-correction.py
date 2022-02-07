import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime as dt, timedelta as td
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from scipy.ndimage import gaussian_filter as gf

np.set_printoptions(precision=3, suppress=True)

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
	

gl_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/glofas/"



station_ids= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']

for station_id in station_ids:
	
	#read in observational data
	df = pd.read_csv("flow.csv", parse_dates=[0,],
                	 date_parser=lambda t:pd.to_datetime(str(t), format='%Y-%m-%dT%H'))

	#read in station metadata
	df_locs = pd.read_csv("station_data.csv")

	#bin missing values, convert data to consistent dtype
	df = df.dropna(subset=[station_id,])
	inflow = df['{}'.format(station_id)].values.astype(float) #convert from cusec to cumet
	dates = df['DateTime'].values



	gl_flow = []



	locx, locy = df_locs[df_locs['LocationID']==station_id][['Longitude', 'Latitude']].values.T
	locx = locx[0]
	locy = locy[0]

	print(locx, locy)


	outdates = []
	obs_flow = []
	s=3 #radius of glofas data to ingest

	#collect observation and nearby glofas data and put into array with consistent 3-hr timesteps
	for date, obs in zip(dates,inflow):
		d = pd.to_datetime(date)
		t = dt(d.year,d.month,d.day)
		t_glo = t+td(days=1)
		#train over finite period because some earlier data is a bit iffy on QC
		if t>dt(2018,6,1): continue
		if t<dt(2005,6,1): continue
		print(station_id, d, obs,)

		outdates.append(dt(d.year, d.month, d.day, d.hour))

		#collect glofas from following day, as it relays discharge from previous 24 hours
		gl_fname = gl_dir + t_glo.strftime("CEMS_ECMWF_dis24_%Y%m%d_glofas_v2.1.nc")
		gl_file = Dataset(gl_fname)

		lons = gl_file.variables['lon'][:] 
		lats = gl_file.variables['lat'][:]
		ix, iy = find(lons, locx), find(lats, locy)

		dis = gl_file.variables['dis24'][0, iy-s:iy+s, ix-s:ix+s]*(3.28084**3)

		gl_flow.append(dis)
		obs_flow.append(obs)


	hours = np.array([t.hour for t in outdates])

	gl_flow_all = np.array(gl_flow)
	Tl, Yl, Xl = np.shape(gl_flow_all)

	gl_flow_central = np.array(gl_flow)[:,s,s]
	obs_flow = np.array(obs_flow)




	#cdf-driven bias adjustment
	obs_quantile_by_hour = {}
	quantiles = np.linspace(0,1,10000)

	for hour in [0,3,6,9,12,15,18,21]:
		prevhour = (hour-3)%24
		nexthour = (hour+3)%24
		prev2hour = (hour-6)%24
		next2hour = (hour+6)%24
		
		obs_subset = obs_flow#[np.in1d(hours,[prevhour,hour,nexthour])]
		_ = np.quantile(obs_subset, quantiles)
		obs_quantile_by_hour[hour] = np.quantile(obs_subset, quantiles)

		np.save("bias_matrices/{}_obs_quantile_{:02d}".format(station_id,hour), obs_quantile_by_hour[hour])


	gl_adjusted_all = np.zeros_like(gl_flow_all)

	gl_sorted = np.zeros_like(gl_flow_all)
	gl_quantile_sorted = np.zeros_like(gl_flow_all)


	for j in range(Yl):
		for i in range(Xl):
			gl_ = gl_flow_all[:,j,i]
			#compute timeline of quantiles for each pixel in the grid
			gl_quantiles = np.array([np.mean(gl_<f) for f in gl_])

			#put these aside so they can be saved and reused later
			gl_sorted[:,j,i] = np.sort(gl_)
			gl_quantile_sorted[:,j,i] = np.sort(gl_quantiles)

			adjustments_by_hour = np.zeros_like(gl_)

			#now do a cdf-match for each of the 3-hour blocks of the day, to simulate a diurnal cycle
			for hour in [0,3,6,9,12,15,18,21]:
				bench = obs_quantile_by_hour[hour]
				gl_sub = gl_quantiles[hours==hour]
				#convert between glofas quantiles and observed quantiles - CDF matching
				gl_adjusted = interp1d(quantiles, bench)(gl_sub)
				adjustments_by_hour[hours==hour] = gl_adjusted

			gl_adjusted_all[:,j,i] = adjustments_by_hour


	np.save("bias_matrices/{}_glofas_quantile_mapping".format(station_id), [gl_sorted, gl_quantile_sorted])

	#extract 3-hourly bias-adjusted GloFAS for observation location
	gl_adjusted_central =  gl_adjusted_all[:,s,s]



	#''' #rejected method, results not as good as the KG-optimised technique
	def linear_superfit(x, *matrix):
		reshaped_matrix = np.reshape(matrix, (1,2*s,2*s))
		#reshaped_matrix = gf(reshaped_matrix, (0, 1,1), mode='constant', cval=0)
		fitted = x*reshaped_matrix[:, :]
		return np.sum(fitted, axis=(-1,-2))

	p0 = np.zeros((2*s,2*s))
	p0[s,s]=1
	#popt, pcov = curve_fit(linear_superfit, gl_adjusted_all, obs_flow, p0=p0.ravel(), bounds=(0,1))
	#np.save("bias_matrices/{}_mse_popt".format(station_id), popt)

	#print(popt.reshape((2*s,2*s)))
	#print("\n")
	#gl_spatially_adjusted = linear_superfit(gl_adjusted_all, *popt)
	#'''



	#find weight matrix to multiply to glofas grid to maximise KG+NS score

	def kg_optimise(*matrix, return_fit=False):
		reshaped_matrix = np.reshape(matrix, (1,2*s,2*s))
		
		#reshaped_matrix = gf(reshaped_matrix, (0, 1,1), mode='constant', cval=0)
		fitted = np.sum(gl_adjusted_all*reshaped_matrix[:, :], axis=(-1,-2))

		#don't want negative values of streamflow!
		fitted = np.clip(fitted, a_min=0, a_max=None)
		if return_fit: return fitted
		score = -kling_gupta(obs_flow,fitted)-nash_sutcliffe(obs_flow,fitted)
		return score

	opt_bounds = [(0,1) for _ in p0.ravel()]
	#opt_bounds[s+2*s*s]=(0.5,1)
	
	res = minimize(kg_optimise, p0.ravel(), bounds=opt_bounds)
	np.save("bias_matrices/{}_kgo_popt".format(station_id), res.x)

	print(res.x.reshape((2*s,2*s)))


	gl_kg_optimised = kg_optimise(res.x, return_fit=True)





	#print("raw KG:{:1.3f} NS:{:1.3f}".format(kling_gupta(obs_flow, gl_flow_central), nash_sutcliffe(obs_flow, gl_flow_central)))
	#print("hcdf KG:{:1.3f} NS:{:1.3f}".format(kling_gupta(obs_flow, gl_adjusted_central), nash_sutcliffe(obs_flow, gl_adjusted_central)))
	#print("hcdf+spat KG:{:1.3f} NS:{:1.3f}".format(kling_gupta(obs_flow, gl_spatially_adjusted), nash_sutcliffe(obs_flow, gl_spatially_adjusted)))
	#print("hcdf+opt KG:{:1.3f} NS:{:1.3f}".format(kling_gupta(obs_flow, gl_kg_optimised), nash_sutcliffe(obs_flow, gl_kg_optimised)))

	plt.figure(figsize=(15,4))
	plt.plot(outdates, obs_flow, 'k-', label="obs")
	plt.plot(outdates, gl_flow_central, 'r-', label="raw glofas KG:{:1.3f} NS:{:1.3f}".format(kling_gupta(obs_flow, gl_flow_central), nash_sutcliffe(obs_flow, gl_flow_central)))
	plt.plot(outdates, gl_adjusted_central, 'b-', label="hcdf-adjusted glofas KG:{:1.3f} NS:{:1.3f}".format(kling_gupta(obs_flow, gl_adjusted_central), nash_sutcliffe(obs_flow, gl_adjusted_central)))
	plt.plot(outdates, gl_kg_optimised, 'g-', label="KG-optimised glofas KG:{:1.3f} NS:{:1.3f}".format(kling_gupta(obs_flow, gl_kg_optimised), nash_sutcliffe(obs_flow, gl_kg_optimised)))

	plt.legend(loc='best')

	plt.title(df_locs[df_locs['LocationID']==station_id]['Description'].values[0])
	plt.ylabel('Flow (cusec)')


	plt.savefig("figs/{}_broadbandbias.png".format(station_id))
	plt.clf()




