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
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree

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
        for p in self.adjacencies_by_point_index:  # add nodes first
            g.add_node(p)
        for p, neighs in self.adjacencies_by_point_index.items():  # now put edges between them
            for p1 in neighs:
                g.add_edge(p, p1)
        return g

    def n_points(self):
        return len(self.adjacencies)

    def get_random_point_index(self):
        return random.randrange(len(self.points))

    def get_neighbors(self, p_i):
        return self.adjacencies_by_point_index[p_i]

    def closest_point_to(self, usp):
        assert type(usp) is UnitSpherePoint
        xyz_as_one_sample = np.array([usp.get_coords("xyz"),])
        indices = self.kdtree.query(xyz_as_one_sample, k=1, return_distance=False)
        # print("got closest point indices:", indices)
        assert type(indices) in [list, np.ndarray]
        assert len(indices) == 1  # scikit-learn return type from these queries
        index_array = indices[0]
        assert index_array.shape == (1,)
        index = index_array[0]
        point_xyz = tuple(self.kdtree.data[index])
        point_number = self.xyz_to_point_number[point_xyz]
        usp = self.points[point_number]
        return usp

    def get_random_path(self, a, b, points_to_avoid):
        # start and end should inch toward each other
        i = 0
        points_in_path = {a, b}
        current_a = a
        current_b = b
        while True:
            which_one = i % 2
            current_point = [current_a, current_b][which_one]
            objective = [current_b, current_a][which_one]
            points_to_avoid_this_step = points_to_avoid | points_in_path
            
            next_step = self.get_next_step_in_path(current_point, objective, points_to_avoid_this_step)
            if which_one == 0:
                current_a = next_step
            elif which_one == 1:
                current_b = next_step
            else:
                raise

            points_in_path.add(next_step)

            if current_a in self.get_neighbors(current_b):
                break

            i += 1
        return points_in_path

    def plot_points(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ps = self.points
        ps_xyz = [p.get_coords("xyz") for p in ps]
        xs = [p[0] for p in ps_xyz]
        ys = [p[1] for p in ps_xyz]
        zs = [p[2] for p in ps_xyz]
        print("x from {} to {}\ny from {} to {}\nz from {} to {}".format(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
        ax.scatter(xs, ys, zs)
        plt.show()

    def place_random_data(self):
        key_str = "x"
        data = {p_i: {key_str: 0} for p_i in range(len(self.points))}

        n_steps = 10
        for i in range(n_steps):
            print("step {}/{}".format(i, n_steps))
            n_patches = random.randint(100, 2000)  # usual: 1000
            area_proportion_per_patch = 1/random.randint(5, 100)  # usual: 1/100
            data = self.add_random_data_patches(data, key_str, n_patches=n_patches, area_proportion_per_patch=area_proportion_per_patch)
        return data

    def add_random_data_patches(self, data, key_str, n_patches, area_proportion_per_patch):
        size_per_patch = int(area_proportion_per_patch * self.n_points())
        print("n_patches: {}; area proportion: 1/{}".format(n_patches, 1/area_proportion_per_patch))
        for i in range(n_patches):
            if i % 100 == 0:
                print("i = {}/{}".format(i, n_patches))
            starting_point = random.choice(self.points)
            # print("starting point: {}".format(starting_point))
            patch = {starting_point}
            # the outward-moving edge is the next points that are not yet in the patch
            edge = set(self.adjacencies[starting_point])
            for p_i in range(size_per_patch):
                chosen = random.choice(list(edge))
                # print("chosen: {}".format(chosen))
                patch.add(chosen)
                edge |= set(self.adjacencies[chosen])
                edge -= patch
                if len(edge) == 0:  # can happen if whole lattice is in patch
                    break
            # change elevation on the patch
            d_el = random.uniform(-100, 100)
            # might want to put the data in a Pandas DataFrame later, for ease of doing stuff like this
            for p in patch:
                p_i = self.usp_to_index[p]
                data[p_i][key_str] += d_el
        return data

    def plot_data(self, data_dict, key_str, size_inches=None, cmap=None, equirectangular=False):
        data_point_indices = list(data_dict.keys())
        data_points = [self.points[p_i] for p_i in data_point_indices]
        latlons_deg = [p.get_coords("latlondeg") for p in data_points]
        lats_deg = np.array([ll[0] for ll in latlons_deg])
        lons_deg = np.array([ll[1] for ll in latlons_deg])
        vals = np.array([data_dict[p_i].get(key_str, 0) for p_i in data_point_indices])
        # print("lat range {} to {}\nlon range {} to {}".format(min(lats_deg), max(lats_deg), min(lons_deg), max(lons_deg)))
        # plt.scatter(lats_deg, lons_deg)
        # plt.show()
        min_val = min(vals)
        max_val = max(vals)
        
        lat_0s = [[   0,    0,    0,   90], [   0,    0,    0,  -90]]
        lon_0s = [[-120,  -60,    0,    0], [  60,  120,  180,    0]]
        n_rows = len(lon_0s)
        n_cols = len(lon_0s[0])
        fig = plt.figure(figsize=size_inches)
        if cmap is None:
            # default to showing elevation
            cmap = pu.get_land_and_sea_colormap()
            contour_levels = pu.get_contour_levels(min_val, max_val, prefer_positive=True)
        else:
            contour_levels = pu.get_contour_levels(min_val, max_val, prefer_positive=False)

        # debugging: print contour levels and colors
        # for level_i in range(len(contour_levels)):
        #     if level_i > 0:
        #         # show the first halfway value as well, just skip it for min (level_i=0)
        #         previous_level_value = contour_levels[level_i - 1]
        #         current_level_value = contour_levels[level_i]
        #         halfway_value_linear = (previous_level_value + current_level_value) / 2
        #         halfway_level_01 = (halfway_value_linear - contour_levels[0]) / (contour_levels[-1] - contour_levels[0])
        #         print("cmap at contour FILL level {} = value {} = RGBA {}".format(level_i - 0.5, halfway_value_linear, cmap(halfway_level_01)))
        # 
        #     level_value = contour_levels[level_i]
        #     level_01 = (level_value - contour_levels[0]) / (contour_levels[-1] - contour_levels[0])
        #     print("cmap at contour LINE level {} = value {} = RGBA {}".format(level_i, level_value, cmap(level_01)))

        if equirectangular:
            x, y = lons_deg, lats_deg  # actual data
            z = vals
            # how many x/y to put on rectangle projection
            n_grid_x = 1000
            n_grid_y = 500
            xi = np.linspace(-180, 180, n_grid_x)
            yi = np.linspace(-90, 90, n_grid_y)
            triang = tri.Triangulation(x, y)
            interpolator = tri.LinearTriInterpolator(triang, z)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)
            # plt.contour(xi, yi, zi, levels=contour_levels, linewidths=0.5, colors='k')
            plt.contourf(xi, yi, zi, levels=contour_levels, cmap=cmap)
        else:
            for i, row in enumerate(lon_0s):
                for j, lon_0 in enumerate(row):
                    lat_0 = lat_0s[i][j]
                    nth_plot = i*len(row) + j + 1
                    ax = fig.add_subplot(n_rows, n_cols, nth_plot)
                    # m = Basemap(projection="cyl")
                    m = Basemap(projection="ortho", lat_0=lat_0, lon_0=lon_0, resolution='l')
                    m.drawmeridians(np.arange(0,360,30))
                    m.drawparallels(np.arange(-90,90,30))
                    MC = m.contourf(lons_deg, lats_deg, vals, levels=contour_levels, cmap=cmap, ax=ax, tri=True, latlon=True)  # latlon=True interprets first two args as LON and LAT RESPECTIVELY
                    # m.contour(lons_deg, lats_deg, vals, levels=[min(vals), 0, max(vals)], colors="k", ax=ax, tri=True, latlon=True)
        
                    # parallel_labels_bools = [1, 1, 0, 0]  # are labels placed at [left right top bottom]
                    # meridian_labels_bools = [0, 0, 1, 1]  # are labels placed at [left right top bottom]
                    # m.drawparallels(np.arange(-90,91,30), labels=parallel_labels_bools)
                    # m.drawmeridians(np.arange(-180,181,30), labels=meridian_labels_bools)
        
                    clb = plt.colorbar(MC, ax=ax)  # without these args, it will say it can't find a mappable object for colorbar
                    clb.ax.set_title(key_str)
                    plt.title("latlon {},{}".format(lat_0, lon_0))


