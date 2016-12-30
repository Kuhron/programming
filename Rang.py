import math
import random

inf = float("inf")
real_range = (-1*inf,inf)

def in_range(range_tuple,value):
	if len(range_tuple) != 2 or range_tuple[0] > range_tuple[1]:
		raise ValueError("Invalid range tuple")
	return value >= range_tuple[0] and value <= range_tuple[1]

def get_random_latitude_and_longitude(degrees=False,lat_range=real_range,lon_range=real_range):
    lat,lon = None,None
    while lat is None or lon is None or (lat_range is not None and not in_range(lat_range,lat)) \
        or (lon_range is not None and not in_range(lon_range,lon)):
	    lon = random.uniform(-1,1)*(180 if degrees else math.pi)
	    cos_theta = random.uniform(-1,1)
	    theta = (math.acos(cos_theta) - math.pi * 1.0/2)
	    lat = theta * (90/(math.pi*1.0/2) if degrees else 1)

	    #print(lat,lon)

    return (lat,lon)

def debug():
	import matplotlib.pyplot as plt

	n = 10**4
	lats = []
	lons = []
	for i in range(n):
		lat,lon = get_random_latitude_and_longitude(degrees=True)
		lats.append(lat)
		lons.append(lon)

	plt.scatter(range(n),lats)
	plt.show()
	plt.close()

	plt.hist(lats)
	plt.show()
	plt.close()

	plt.scatter(range(n),lons)
	plt.show()
	plt.close()

	plt.hist(lons)
	plt.show()
	plt.close()

mode = input("Select mode:\n1. World\n2. Continental US (approx.)\n")
if mode == "1":
	print(get_random_latitude_and_longitude(degrees=True))
elif mode == "2":
	print(get_random_latitude_and_longitude(degrees=True,lat_range=(24.5,49.5),lon_range=(-125,-66)))














