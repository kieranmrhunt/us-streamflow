import datetime
import subprocess
import json
import os, sys
import netCDF4 as nc
from ecmwfapi import ECMWFService
server = ECMWFService("mars", url="https://api.ecmwf.int/v1", key="userkey", email="useremail")

 
def mars_request(idate):
    """      
        A MARS request for all variables listed in variables dictionary.
        Returns a netcdf file for each day for each variable.
    """
    
    path = '/storage/shared/research/met/bitmap/mr806421/us-rivers/daily-data/'
    
    
    with open(path+ 'variables.json') as jf:
        variables = json.load(jf)
    
    for var_name in list(variables.keys()):
        var_code = variables[var_name]
        for day in range(1,11):
            
            file = path + idate + '_' + var_name + '_day' + str(day) + '.grb'
            #file = path + idate + var_name +'.nc'#+ '_day' + str(day) + '.nc'
            #nc.Dataset(file, 'w', format='NETCDF4')
            nc_file = path + idate + '_' + var_name + '_day' + str(day) + '.nc'
            
            if os.path.isfile(nc_file):
    	        continue


            #stincr = 6
            st1 = (24*(day-1))
            #st2 = (24*day)
            steps = '%i/%i/%i/%i' % (st1, st1 + 6, st1 + 12, st1 + 18)#str(st1 + stincr) + '/to/' + str(st2) + '/by' + str(stincr)
            server.execute({
            "class": "od",
            "date": idate,
            "expver": "1",
            "levtype": "sfc",
            "param": str(var_code),
            "step": steps, 
            "stream": "oper",
            "time": "00",
            "type": "fc",
            "grid": "0.1/0.1",

            }, file)
            os.chmod(file, 0o777)
            os.chmod(file, 0o777)
            subprocess.run(['grib_to_netcdf -D NC_FLOAT %s -o %s' % (file, nc_file)], shell=True)
            subprocess.run(['rm %s' % file], shell=True)

            
if __name__ == '__main__':
	
	idate = sys.argv[1]
	mars_request(idate)
	
		
	
	
	
	
	
	
    
