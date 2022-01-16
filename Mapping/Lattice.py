# base class to inherit from for different lattice types
# e.g. lat/lon or geodesic
# this class should implement the methods that exist regardless of the grid topology
# e.g. finding nearest neighbors, finding all neighbors, converting to xyz, etc.

import random
import string
import os
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri  # interpolation of irregularly spaced data
import numpy as np
import networkx as nx
import pandas as pd
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
from datetime import datetime

from UnitSpherePoint import UnitSpherePoint
import PlottingUtil as pu
import MapCoordinateMath as mcm
import NoiseMath as nm
import IcosahedronMath as ihm


class Lattice:
    def __init__(self):
        random_name = "".join(random.choice(string.ascii_lowercase) for _ in range(12))
        self.project_name = datetime.utcnow().strftime("%Y-%m-%d-%H%M%S") + "-" + random_name

    def get_adjacencies(self):
        # specific to the subclasses, depending on type of lattice
        raise NotImplementedError

    # def get_graph(self):
    #     raise Exception("do not use")
    #     g = nx.Graph()
    #     for p in self.adjacencies_by_point_index:  # add nodes first
    #         g.add_node(p)
    #     for p, neighs in self.adjacencies_by_point_index.items():  # now put edges between them
    #         for p1 in neighs:
    #             g.add_edge(p, p1)
    #     return g

    def get_n_points(self):
        return len(self.points)

    def get_random_point_index(self):
        return random.randrange(len(self.points))

    def get_neighbors(self, p_i):
        return self.adjacencies_by_point_index[p_i]

    def get_coords(self, coord_system=None, point_indices=None):
        # allow getting only a subset of the points so don't have to hold everything in memory all at once
        print("getting coords for {}".format(type(self)))
        xyz_coords = []
        latlondeg_coords = []
        if coord_system == "xyz":
            # python varnames are nametags on objects
            coords = xyz_coords
        elif coord_system == "latlondeg":
            coords = latlondeg_coords
        elif coord_system is None:
            # again, nametag on object, the object should change while this name continues to refer to it
            coords = [xyz_coords, latlondeg_coords]
        else:
            raise ValueError("unknown coordinate system {}".format(coord_system))

        if point_indices is None:
            point_indices = self.get_point_indices()
        for point_number in point_indices:
            if point_number % 1000 == 0:
                print("point number {}/{}".format(point_number, len(point_indices)))
            pos = self.get_position_mathematical(point_number)
            if coord_system is None:
                xyz_coords.append(tuple(pos["xyz"]))
                latlondeg_coords.append(tuple(pos["latlondeg"]))
            else:
                tup = tuple(pos[coord_system])
                coords.append(tup)
        print("done getting coords for {}".format(type(self)))
        return coords

    def get_xyz_coords(self, point_indices=None):
        print("getting xyz_coords for {}".format(type(self)))
        res = np.array(self.get_coords(coord_system="xyz", point_indices=point_indices))
        print("done getting xyz_coords for {}".format(type(self)))
        return res

    def get_latlondeg_coords(self, point_indices=None):
        print("getting latlondeg_coords for {}".format(type(self)))
        res = np.array(self.get_coords(coord_system="latlondeg", point_indices=point_indices))
        print("done getting latlondeg_coords for {}".format(type(self)))
        return res

    def get_kdtree(self):
        print("getting KDTree")
        res = KDTree(self.get_xyz_coords())
        print("done getting KDTree")
        return res

    def get_point_indices(self):
        return list(range(self.n_points))

    def get_position_mathematical(self, point_number):
        raise NotImplementedError("subclass should implement")

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
        return point_number, usp

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

    def create_dataframe(self, point_indices=None, with_coords=False):
        print(f"creating DataFrame for {type(self)}")
        if point_indices is None:
            point_indices = self.get_point_indices()
        df = pd.DataFrame(index=point_indices)
        if with_coords:
            xyz_coords, latlondeg_coords = self.get_coords(point_indices=point_indices)
            df["xyz"] = xyz_coords
            df["latlondeg"] = latlondeg_coords
        print(f"done creating DataFrame for {type(self)}")
        return df

    def place_random_data(self, key_str, df=None):
        if df is None:
            df = self.create_dataframe()
        point_indices = df.index
        if key_str not in df.columns:
            df[key_str] = [0 for p_i in point_indices]
        df = nm.change_globe(df, key_str)
        return df

    def write_data(self, df, output_fp):
        if os.path.exists(output_fp):
            raise IOError("output filepath exists! Aborting. fp = {}".format(output_fp))
        columns_to_exclude = ["usp", "xyz", "latlondeg"]  # coordinate things that can be recalculated or retrieved from memoization files as needed, don't store them in the database
        # Note that by default, .drop() does not operate inplace; despite the ominous name, df is unharmed by this process. (from https://stackoverflow.com/questions/29763620/)
        for col in columns_to_exclude:
            if col in df.columns:  # otherwise it will raise error that the column is not found
                new_df = df.drop(col, axis=1)
                assert new_df is not df, "uh-oh, we edited the df in-place"
                df = new_df
        df.to_csv(output_fp, index_label="index")

    def plot_data(self, df, key_str, size_inches=None, cmap=None, equirectangular=True, save=False, category_labels=None, contour_lines=False):
        data_point_indices = df.index
        # data_points = [self.points[p_i] for p_i in data_point_indices]
        if "latlondeg" in df.columns:
            latlons_deg = df["latlondeg"]
        else:
            point_indices = df.index
            latlons_deg = ihm.get_latlons_from_point_numbers(point_indices)
        lats_deg = np.array([ll[0] for ll in latlons_deg])
        lons_deg = np.array([ll[1] for ll in latlons_deg])
        vals = df[key_str]

        if category_labels is not None:
            # data is categorical, give it integers based on the list index of the category value
            assert set(vals) - set(category_labels) == set(), "vals found: {}, category labels: {}, extra vals not accounted for".format(set(vals), category_labels)
            new_vals = []
            for val in vals:
                new_val_index = category_labels.index(val)
                new_vals.append(new_val_index)
            vals = new_vals

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
            contourf_levels = pu.get_contour_levels(min_val, max_val, prefer_positive=True)
            contour_line_levels = pu.get_contour_levels(min_val, max_val, prefer_positive=True, n_sea_contours=5, n_land_contours=15)
        else:
            contourf_levels = pu.get_contour_levels(min_val, max_val, prefer_positive=False)
            contour_line_levels = contourf_levels

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
            MC = plt.contourf(xi, yi, zi, levels=contourf_levels, cmap=cmap)
            clb = plt.colorbar(MC, ax=plt.gca())  # without these args, it will say it can't find a mappable object for colorbar
            if contour_lines:
                plt.contour(xi, yi, zi, levels=contour_line_levels, linewidths=0.5, colors='k')
            clb.ax.set_title(key_str)
            plt.gca().set_facecolor('k')  # contourf won't draw over this for out-of-range values
        else:
            for i, row in enumerate(lon_0s):
                for j, lon_0 in enumerate(row):
                    lat_0 = lat_0s[i][j]
                    nth_plot = i*len(row) + j + 1
                    ax = fig.add_subplot(n_rows, n_cols, nth_plot)
                    # m = Basemap(projection="cyl")
                    raise Exception("Basemap doesn't work anymore")
                    # m = Basemap(projection="ortho", lat_0=lat_0, lon_0=lon_0, resolution='l')
                    m.drawmeridians(np.arange(0,360,30))
                    m.drawparallels(np.arange(-90,90,30))
                    MC = m.contourf(lons_deg, lats_deg, vals, levels=contourf_levels, cmap=cmap, ax=ax, tri=True, latlon=True)  # latlon=True interprets first two args as LON and LAT RESPECTIVELY
                    # m.contour(lons_deg, lats_deg, vals, levels=[min(vals), 0, max(vals)], colors="k", ax=ax, tri=True, latlon=True)
        
                    # parallel_labels_bools = [1, 1, 0, 0]  # are labels placed at [left right top bottom]
                    # meridian_labels_bools = [0, 0, 1, 1]  # are labels placed at [left right top bottom]
                    # m.drawparallels(np.arange(-90,91,30), labels=parallel_labels_bools)
                    # m.drawmeridians(np.arange(-180,181,30), labels=meridian_labels_bools)
        
                    clb = plt.colorbar(MC, ax=ax)  # without these args, it will say it can't find a mappable object for colorbar
                    clb.ax.set_title(key_str)
                    plt.gca().set_facecolor('k')  # contourf won't draw over this for out-of-range values
                    plt.title("latlon {},{}".format(lat_0, lon_0))
        if save:
            name = "Projects/{}_{}".format(self.project_name, key_str)
            fig_fp = name + ".png"
            plt.savefig(fig_fp)
            data_fp = name + ".txt"
            with open(data_fp, "w") as f:
                f.write("\n".join(str(x) for x in df[key_str]))

