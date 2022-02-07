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

np.set_printoptions(precision=3, suppress=True)

s=3

gl_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/glofas/"
compute_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/compute/"

station_codes= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']
df_locs = pd.read_csv("station_data.csv")



for station_code in station_codes:
	print(station_code)
	
	

	kgo_popt = np.load(compute_dir+"bias_matrices/{}_kgo_popt.npy".format(station_code))

	reshaped_popt = np.reshape(kgo_popt, (2*s,2*s))
	
	print(reshaped_popt)	
