# base class to inherit from for different lattice types
# e.g. lat/lon or geodesic
# this class should implement the methods that exist regardless of the grid topology
# e.g. finding nearest neighbors, finding all neighbors, converting to xyz, etc.

import random
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri  # interpolation of irregularly spaced data
import numpy as np

from UnitSpherePoint import UnitSpherePoint


class Lattice:
    def __init__(self):
        raise NotImplementedError("do not initialize Lattice itself; use a subclass such as LatitudeLongitudeLattice")

    def get_adjacencies(self):
        # specific to the subclasses, depending on type of lattice
        raise NotImplementedError

    def get_points(self):
        return list(self.adjacencies.keys())

    def get_nearest_lattice_point_to_input_point(p):
        assert type(p) is UnitSpherePoint
        # first pass: brute force
        return min(self.adjacencies, key=lambda x: x.distance(p))

        # later could optimize somehow, e.g. take only a box of +/- dx,dy,dz and sort those

    def plot_points(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ps = self.get_points()
        ps_xyz = [p.get_coords("xyz") for p in ps]
        xs = [p[0] for p in ps_xyz]
        ys = [p[1] for p in ps_xyz]
        zs = [p[2] for p in ps_xyz]
        print("x from {} to {}\ny from {} to {}\nz from {} to {}".format(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
        ax.scatter(xs, ys, zs)
        plt.show()

    def place_random_data(self):
        data = {p: 0 for p in self.points}
        n_patches = 1000
        for i in range(n_patches):
            starting_point = random.choice(self.points)
            patch = {starting_point}
            # the outward-moving edge is the next points that are not yet in the patch
            edge = set(self.adjacencies[starting_point])
            while True:
                chosen = random.choice(list(edge))
                patch.add(chosen)
                edge |= set(self.adjacencies[chosen])
                edge -= patch
                if random.random() < 0.01:
                    break
                if len(edge) == 0:  # can happen if whole lattice is in patch
                    break
            # change elevation on the patch
            d_el = random.uniform(-100, 100)
            # might want to put the data in a Pandas DataFrame later, for ease of doing stuff like this
            for p in patch:
                data[p] += d_el

        return data

    def plot_data(self, data):
        data_points = list(data.keys())  # UnitSpherePoint objects
        latlons_deg = [p.get_coords("latlondeg") for p in data_points]
        lats_deg = np.array([ll[0] for ll in latlons_deg])
        lons_deg = np.array([ll[1] for ll in latlons_deg])
        vals = np.array([data[p] for p in data_points])
        # print("lat range {} to {}\nlon range {} to {}".format(min(lats_deg), max(lats_deg), min(lons_deg), max(lons_deg)))
        # plt.scatter(lats_deg, lons_deg)
        # plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        levels = 20
        # m = Basemap(projection="cyl")
        m = Basemap(projection="ortho", lat_0=0., lon_0=0., resolution='l')
        MC = m.contourf(lons_deg, lats_deg, vals, levels, ax=ax, tri=True, latlon=True)  # latlon=True interprets first two args as LON and LAT RESPECTIVELY
        # m.contour(lats_deg, lons_deg, vals, levels=[min(vals), 0, max(vals)], colors="k", ax=ax, latlon=True)

        # parallel_labels_bools = [1, 1, 0, 0]  # are labels placed at [left right top bottom]
        # meridian_labels_bools = [0, 0, 1, 1]  # are labels placed at [left right top bottom]
        # m.drawparallels(np.arange(-90,91,30), labels=parallel_labels_bools)
        # m.drawmeridians(np.arange(-180,181,30), labels=meridian_labels_bools)

        plt.colorbar(MC, ax=ax)  # without these args, it will say it can't find a mappable object for colorbar
        plt.show()

