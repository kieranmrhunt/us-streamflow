import sys, os
from datetime import datetime as dt, timedelta as td

datelist = [dt(2020,8,15) + td(days=i) for i in range(382)]
#datelist = [dt(2021,6,2) + td(days=i) for i in range(382)]

for date in datelist:
	if date<dt(2021,8,7): continue
	
	idate = date.strftime('%Y%m%d')
	print(idate)
	
	os.system("python retrieve_mars_data_IFS.py {}".format(idate))
	
	if date>=dt(2020,9,1):
		
		if not os.path.isfile("/home/users/rz908899/cluster/mr806421/us-rivers/compute/hindcast-catchment-data/NFSW4{}T00.csv".format(idate)):
			os.system("python operational-make-catchment-file.py {}".format(idate))
		
		cutoff_date = date-td(days=14)
		cutoff_idate = cutoff_date.strftime('%Y%m%d')
		os.system("rm /home/users/rz908899/cluster/mr806421/us-rivers/daily-data/{}*_day*.nc".format(cutoff_idate))

	
	
	
