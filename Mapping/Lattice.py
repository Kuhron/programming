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
import networkx as nx

from UnitSpherePoint import UnitSpherePoint
import PlottingUtil as pu


class Lattice:
    def __init__(self):
        raise NotImplementedError("do not initialize Lattice itself; use a subclass such as LatitudeLongitudeLattice")

    def get_adjacencies(self):
        # specific to the subclasses, depending on type of lattice
        raise NotImplementedError

    def get_graph(self):
        g = nx.Graph()
        for p in self.adjacencies:  # add nodes first
            g.add_node(p)
        for p, neighs in self.adjacencies.items():  # now put edges between them
            for p1 in neighs:
                g.add_edge(p, p1)
        return g

    def get_points(self, coords_system=None):
        points = list(self.adjacencies.keys())
        if coords_system is not None:
            return [p.get_coords(coords_system) for p in points]
        return points

    def n_points(self):
        return len(self.adjacencies)

    def get_random_point(self):
        return random.choice(list(self.adjacencies.keys()))

    def closest_point_to(self, p):
        assert type(p) is UnitSpherePoint

        x, y, z = p.get_coords("xyz")
        d_xyz = 0.1
        is_candidate = lambda x1, y1, z1: abs(x-x1) < d_xyz and abs(y-y1) < d_xyz and abs(z-z1) < d_xyz
        ps = self.get_points()
        candidates = [p1 for p1 in ps if is_candidate(*p1.get_coords("xyz"))]

        return min(candidates, key=lambda x: x.distance(p))

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
        size_per_patch = int(1/100 * self.n_points())
        for i in range(n_patches):
            starting_point = random.choice(self.points)
            patch = {starting_point}
            # the outward-moving edge is the next points that are not yet in the patch
            edge = set(self.adjacencies[starting_point])
            for p_i in range(size_per_patch):
                chosen = random.choice(list(edge))
                patch.add(chosen)
                edge |= set(self.adjacencies[chosen])
                edge -= patch
                if len(edge) == 0:  # can happen if whole lattice is in patch
                    break
            # change elevation on the patch
            d_el = random.uniform(-100, 100)
            # might want to put the data in a Pandas DataFrame later, for ease of doing stuff like this
            for p in patch:
                data[p] += d_el

        return data

    def plot_data(self, data, size_inches=None):
        data_points = list(data.keys())  # UnitSpherePoint objects
        latlons_deg = [p.get_coords("latlondeg") for p in data_points]
        lats_deg = np.array([ll[0] for ll in latlons_deg])
        lons_deg = np.array([ll[1] for ll in latlons_deg])
        vals = np.array([data[p] for p in data_points])
        # print("lat range {} to {}\nlon range {} to {}".format(min(lats_deg), max(lats_deg), min(lons_deg), max(lons_deg)))
        # plt.scatter(lats_deg, lons_deg)
        # plt.show()
        min_elevation = min(vals)
        max_elevation = max(vals)
        
        lon_0s = [[-120, -60, 0], [60, 120, 180]]
        n_rows = len(lon_0s)
        n_cols = len(lon_0s[0])
        fig = plt.figure(figsize=size_inches)
        cmap = pu.get_land_and_sea_colormap()
        contour_levels = pu.get_contour_levels(min_elevation, max_elevation)
        for i, row in enumerate(lon_0s):
            for j, lon_0 in enumerate(row):
                nth_plot = i*len(row) + j + 1
                ax = fig.add_subplot(n_rows, n_cols, nth_plot)
                # m = Basemap(projection="cyl")
                m = Basemap(projection="ortho", lat_0=0., lon_0=lon_0, resolution='l')
                MC = m.contourf(lons_deg, lats_deg, vals, levels=contour_levels, cmap=cmap, ax=ax, tri=True, latlon=True)  # latlon=True interprets first two args as LON and LAT RESPECTIVELY
                # m.contour(lons_deg, lats_deg, vals, levels=[min(vals), 0, max(vals)], colors="k", ax=ax, tri=True, latlon=True)
    
                # parallel_labels_bools = [1, 1, 0, 0]  # are labels placed at [left right top bottom]
                # meridian_labels_bools = [0, 0, 1, 1]  # are labels placed at [left right top bottom]
                # m.drawparallels(np.arange(-90,91,30), labels=parallel_labels_bools)
                # m.drawmeridians(np.arange(-180,181,30), labels=meridian_labels_bools)
    
                plt.colorbar(MC, ax=ax)  # without these args, it will say it can't find a mappable object for colorbar
                plt.title("lon {}".format(lon_0))

