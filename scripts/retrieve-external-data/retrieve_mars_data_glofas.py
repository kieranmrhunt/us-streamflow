import datetime
import json
import os
import subprocess
import netCDF4 as nc
from ecmwfapi import ECMWFService
server = ECMWFService("mars", url="https://api.ecmwf.int/v1", key="userkey", email="useremail")

def glofas_mars_request(idate,model):
    """
    A MARS request for GloFAS control forecast of discharge.
    Returns a netcdf file containing all 10 days worth of forecasts.
    """

    path = '/storage/shared/research/met/bitmap/mr806421/us-rivers/daily-data/'
    nc_file = path + idate + '_dis.nc'
    if os.path.isfile(nc_file):
    	return
    
    grb_file = path + idate + '_dis.grb'
    server.execute({
    "class":"ce",
    "date":idate,
    "expver":"1",
    "levtype":"sfc",
    "model":model,
    "origin":"ecmf",
    "param":"240024",
    "step":"24/48/72/96/120/144/168/192/216/240/264/288",
    "stream":"wfas",
    "time":"00:00:00",
    "type":"cf",
    "grid": "0.1/0.1",
    #"format": "netcdf"
    } , grb_file  )
    os.chmod(grb_file,0o777)
    subprocess.run(['grib_to_netcdf -D NC_FLOAT %s -o %s' % (grb_file, nc_file)], shell=True)
    subprocess.run(['rm %s' % grb_file], shell=True)
        
if __name__ == '__main__':
    datelist = [datetime.datetime(2020,8,15) + datetime.timedelta(days=i) for i in range(365)]
    
    for date in datelist:
    	print(date)
    	if date<=datetime.datetime(2021,5,25):
    		model = 'lisflood'
    	else:
    		model = "global_lisflood"
    		
    	idate = date.strftime('%Y%m%d')
    	glofas_mars_request(idate, model)
  
