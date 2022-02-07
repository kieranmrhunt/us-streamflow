import numpy as np
from datetime import datetime as dt, timedelta as td
from netCDF4 import Dataset, num2date
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter as gf

gl_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/daily-data/"
compute_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/compute/"
find = lambda x, arr: np.argmin(np.abs(x-arr))

for start_date in [dt(2020,9,1)+td(days=i) for i in range(395)]:
	
	fc_times = [start_date+td(hours=18+i*6) for i in range(40)]
	
	handle_raw = 'TC+KHGMEC_raw'
	
	df_locs = pd.read_csv(compute_dir+"station_data.csv")

	gl_fname = gl_dir + start_date.strftime("%Y%m%d_dis.nc")
	gl_file = Dataset(gl_fname)
	gl_tobj = gl_file.variables['time']
	gl_dates = num2date(gl_tobj[:], gl_tobj.units)
	print(gl_dates)
	
	
	
	
	break	
	







