import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from IcosahedronPointDatabase import IcosahedronPointDatabase
import IcosahedronPointDatabase as icdb
import IcosahedronMath as icm
import PlottingUtil as pu
from XyzLookupAncestryGraph import XyzLookupAncestryGraph


def plot_variable_whole_world(db, variable_name, xyzg):
    lns = db.get_all_lookup_numbers()
    lns = random.sample(list(lns.values), 100000)
    pu.plot_variable_interpolated_from_db(db, lns, variable_name, xyzg, resolution=1000, show=False)
    plt.gcf().set_size_inches(18, 6)
    now_str = datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
    plt.savefig(f"ElevationImages/WorldMap_{variable_name}_{now_str}.png")
    plt.gcf().clear()

    # ideally it would do resolution per degree (since equirectangular should have twice as many longitude pixels as latitude, but the resolution kwarg doesn't know that)
    # also, might be better to make the grid and interpolate at those points by nearest neighbor (within tolerance distance, beyond which there should be no value plotted, so we don't get things in the middle of the ocean with false data plotted there because the nearest neighbor is 1000 miles away)
    # to do nearest neighbor, or something like that, DON'T use an actual nearest neighbor algorithm because it will be querying the whole planet, instead do the function that finds points within a circle region, and make the radius the tolerance, and then find nearest neighbor among those (or average their values or something like that) to get the value to plot at this point


if __name__ == "__main__":
    db_root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"
    db = IcosahedronPointDatabase.load(db_root_dir)
    planet_radius_km = icm.CADA_II_RADIUS_KM
    xyzg = XyzLookupAncestryGraph()  # will add to it as needed

    plot_variable_whole_world(db, "elevation", xyzg)

