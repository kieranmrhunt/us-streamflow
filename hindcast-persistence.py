import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt, timedelta as td


compute_dir = "/storage/shared/research/met/bitmap/mr806421/us-rivers/compute/"

station_codes= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']

lead_times = 1,

for lead_time in lead_times:

	for start_date in [dt(2020,9,1)+td(days=i) for i in range(395)]:
		
		ndays = 10 #length of forecast
		datelist = [start_date+td(hours=18) + i*td(hours=6) for i in range(0,4*ndays)]

		handle = 'TC+KHGMEC_lstm'
		fname = start_date.strftime('hindcasts/persistence_%Y%m%dT%H.csv'.format(lead_time))

		if os.path.isfile(fname):
			continue

		f = open(fname, 'w+')
		f.write('DateTime,LocationID,ForecastTime,VendorID,Value,Units\n')


		fcast_dstring = start_date.strftime('%Y-%m-%dT%H')
		f_dstring = start_date.strftime('%Y%m%dT%H')

		gt_df = pd.read_csv(compute_dir+'recent_obs.csv')
		
		fcast_date = start_date - td(days=int(lead_time))
		


		for station_code in station_codes:
			print(start_date, station_code)
			
			gt = gt_df[gt_df.LocationID==station_code]
			gt_dates = np.array([dt.strptime(t, "%Y-%m-%dT%H") for t in gt.DateTime.values])
			
			
			itg = np.searchsorted(gt_dates, fcast_date)
			
			gt_vals = gt.Value.values[itg-16:itg]

			if 'Ice' in gt_vals:
				gt_val = 0
			else:
				gt_val = np.nanmean(gt_vals.astype(float))
			
			pers_flow = np.ones_like(datelist)*gt_val
			pers_flow = np.nan_to_num(pers_flow, posinf=0, neginf=0).astype(float)
			
			for datum, date in zip(pers_flow, datelist):
				if date not in datelist: continue
				dstring = date.strftime("%Y-%m-%dT%H")
				f.write('{},{},{},{},{:1.2f},CFS\n'.format(dstring, station_code, fcast_dstring, handle, datum))
			


		f.close()






