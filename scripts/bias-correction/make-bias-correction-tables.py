import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta as td

station_ids= ['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']

station_id = 'ARWN8'

#df_locs = pd.read_csv(compute_dir+"station_data.csv")

historic = pd.read_csv("catchment-data/{}.csv".format(station_id)).dropna()
historic_obs = historic['inflow_obs'].values
historic_glf = historic['glofas_s'].values

R = 0.1
q = np.mean(historic_glf<=R)
Q = np.nanquantile(historic_obs, q)

print(R,q,Q)

op_datelist = []
start = dt(2020,10,21)
end = dt.today()-td(days=1)
while start<end:
	op_datelist.append(start)
	start+=td(hours=6)

f_datelist = []
start = dt(2020,10,21)
end = dt.today()-td(days=1)
while start<end:
	op_datelist.append(start)
	start+=td(days=1)

op_datelist, f_datelist = np.array(op_datelist), np.array(f_datelist)

forecasts = {"D{}".format(i):np.ones_like(op_datelist)* for i in range(1,11)}



for f_date in f_datelist:
	fdstring = f_date.strftime("%Y%m%dT%H")
	fname = "operational-catchment-data/{}{}.csv".format(station_id,fdstring)
	print(fname)
	df = pd.read_csv(fname)
	
	break

#20201021T00.csv




