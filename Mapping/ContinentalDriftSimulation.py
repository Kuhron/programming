# make a set of UnitSpherePoints, each labeled as either continental or oceanic (maybe a "density" variable")
# move them around in random directions (maybe assign them to groups if calculating the motion of each point in the group is not too hard)
# each also has an "elevation" variable
# when collision occurs, cont+cont makes both of them go up, oc+oc also makes them both go up?, cont+oc makes cont go up and oc go down


import random
import numpy as np
import matplotlib.pyplot as plt

from UnitSpherePoint import UnitSpherePoint
import PlottingUtil as pu


def get_initial_usps(n_points):
    res = []
    for i in range(n_points):
        usp = UnitSpherePoint.random()
        density = random.choice([1,2])
        assert not hasattr(usp, "density")
        assert not hasattr(usp, "elevation")
        usp.density = density
        usp.elevation = random.random()
        res.append(usp)
    return res


def plot_elevations(usps):
    data_coords = [usp.latlondeg() for usp in usps]
    values = [usp.elevation for usp in usps]
    lat_range = [-90, 90]
    lon_range = [-180, 180]
    n_lats, n_lons = [1000, 2000]
    pu.plot_interpolated_data(data_coords, values, lat_range, lon_range, n_lats, n_lons, with_axis=True)
    plt.savefig("ContinentalDriftElevation.png")
    plt.gcf().clear()


if __name__ == "__main__":
    usps = get_initial_usps(1000)
    plot_elevations(usps)
