import datetime
import requests
import traceback
from pydap.client import open_url

dt = datetime.datetime(2014, 7, 1, 0, 0, 0)
ym_str = dt.strftime("%Y%m")
ymd_str = dt.strftime("%Y%m%d")

# url = 'http://nomads.ncdc.noaa.gov/dods/NCEP_NARR_DAILY/197901/197901/narr-a_221_197901dd_hh00_000'  # doctype, because this refers to the info page
url = "https://nomads.ncdc.noaa.gov/dods/NCEP_NARR_DAILY/{0}/{1}/narr-a_221_{1}_0000_000".format(ym_str, ymd_str)  # works! probably earliest dates have problems
# url = 'http://test.opendap.org/dap/data/nc/coads_climatology.nc'  # sometimes works, sometimes 500 error
# url = 'http://dapper.pmel.noaa.gov/dapper/argo/argo_all.cdp'  # doctype, because it actually returns an HTML error page
# url = 'http://nomads.ncdc.noaa.gov/thredds/dodsC/nam218/201111/20111102/nam_218_20111102_0000_000.grb'  # doctype, because returns an error page
# url = "http://nomads.ncep.noaa.gov:9090/dods/nam/nam20111206/nam1hr_00z"  # error page


try:
    data = open_url(url)
    print(type(data))
except Exception as e:
    traceback.print_tb(e.__traceback__)
    print("----")
    page = requests.get(url)
    print(page)
    print(page.text[:1000])

# print(data.keys())
tmp2m = data['tmp2m']
print(type(tmp2m))
print(tmp2m.keys())
print(tmp2m[:, 0, 0])

# lat_index = 200    # you could tie this to tmp2m.lat[:]
# lon_index = 200    # you could tie this to tmp2m.lon[:]
# print(tmp2m.array[:,lat_index,lon_index])
