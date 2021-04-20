import numpy as np
import random
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree

from Lattice import Lattice
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm
import DeterminantInsideTriangle as det_in_tri


class LatitudeLongitudeLattice(Lattice):
    # a rectangular lattice on a part of the globe
    # can be warped to fit the quadrilateral between any four points
    # these points are called p00, p01, p10, p11 (row-column nomenclature)
    # e.g. p00 = Seattle, p01 = NYC, p10 = San Diego, p11 = Miami

    def __init__(self, r_size, c_size, latlon00, latlon01, latlon10, latlon11):
        super().__init__()  # make project name
        self.r_size = r_size
        self.c_size = c_size
        # self.x_range = np.arange(self.r_size)
        # self.y_range = np.arange(self.c_size)
        self.lat00, self.lon00 = latlon00
        self.lat01, self.lon01 = latlon01
        self.lat10, self.lon10 = latlon10
        self.lat11, self.lon11 = latlon11

        self.xyz00 = mcm.unit_vector_lat_lon_to_cartesian(self.lat00, self.lon00, deg=True)
        self.xyz01 = mcm.unit_vector_lat_lon_to_cartesian(self.lat01, self.lon01, deg=True)
        self.xyz10 = mcm.unit_vector_lat_lon_to_cartesian(self.lat10, self.lon10, deg=True)
        self.xyz11 = mcm.unit_vector_lat_lon_to_cartesian(self.lat11, self.lon11, deg=True)

        # self.create_point_dicts()
        self.n_points = self.get_n_points()
        # self.adjacencies = self.get_adjacencies()
        # self.graph = self.get_graph()
        self.xyz_coords = self.get_xyz_coords()
        self.kdtree = self.get_kdtree()  # for distance calculation

    def get_n_points(self):
        return self.r_size * self.c_size

    def get_points(self):
        for latlondeg, xyz in zip(self.get_latlondegs(), self.get_xyzs()):
            usp = UnitSpherePoint({"latlondeg": latlondeg, "xyz": xyz})
            yield usp

    def get_rc_generator(self):
        print("- getting rc tuples for {}".format(type(self)))
        for r in range(self.r_size):
            if r % 10 == 0:
                print("row {}/{}".format(r, self.r_size))
            for c in range(self.c_size):
                yield (r, c)
        print("- done getting rc tuples for {}".format(type(self)))

    def get_rc_meshgrid(self):
        # NOTE: need to do it column first! since numpy treats the column number as x coordinate
        # otherwise it will be transposed
        cols, rows = np.meshgrid(range(self.c_size), range(self.r_size))
        # NOW need to return rows first!
        return rows, cols

    def get_rc_array(self):
        rows, cols = self.get_rc_meshgrid()
        res = np.stack(rows, cols)
        assert res.shape == (2, self.r_size, self.c_size)  # point-array-last form
        return res

    def get_latlondegs_generator(self):
        for r, c in self.get_rc_tuples():
            latlondeg = self.single_rc_to_latlon(r, c)
            yield latlondeg

    def get_latlondegs_array(self):
        rs, cs = self.get_rc_meshgrid()
        res = self.rc_array_to_latlon(rs, cs)
        expected_shape = (2, self.r_size, self.c_size)
        assert res.shape == expected_shape, "{} != {}".format(res.shape, expected_shape)
        return res

    def get_xyzs_generator(self):
        for latlondeg in self.get_latlondegs():
            lat, lon = latlondeg
            xyz = mcm.unit_vector_lat_lon_to_cartesian(lat, lon, deg=True)
            yield xyz

    def get_xyzs_array(self):
        latlondegs = self.get_latlondegs_array()
        lats, lons = latlondegs
        res = mcm.unit_vector_lat_lon_to_cartesian(lats, lons, deg=True)
        expected_shape = (3, self.r_size, self.c_size)
        assert res.shape == expected_shape, "{} != {}".format(res.shape, expected_shape)
        return res

    def get_xyz_coords(self):
        print("- getting xyz coords for {}".format(type(self)))
        arr = self.get_xyzs_array()
        expected_arr_shape = (3, self.r_size, self.c_size)
        assert arr.shape == expected_arr_shape, "{} != {}".format(arr.shape, expected_arr_shape)
        lst = []
        for r, c in self.get_rc_generator():
            arr_item = arr[:,r,c]
            assert len(arr_item) == 3
            lst.append(tuple(arr_item))
        res = np.array(lst)
        print("- done getting xyz coords for {}".format(type(self)))
        return res

    def get_kdtree(self):
        print("- getting KDTree for {}".format(type(self)))
        arr = self.xyz_coords
        n_samples, n_features = arr.shape  # should throw for invalid shape for the KDTree constructor
        assert n_features == 3, n_features
        res = KDTree(arr)
        print("- done getting KDTree for {}".format(type(self)))
        return res

    def get_latlons_array(self):
        latlons_array = self.rc_array_to_latlon(rows, cols)
        return latlons_array

    def single_rc_to_latlon(self, row, col):
        assert type(row) in [int, float]
        assert type(col) in [int, float]
        rows = np.array([row])
        cols = np.array([col])
        latlon_array = self.rc_array_to_latlon(rows, cols)
        assert latlon_array.shape == (2, 1)
        lat_arr, lon_arr = latlon_array
        lat, = lat_arr
        lon, = lon_arr
        return np.array([lat, lon])

    def rc_array_to_latlon(self, rows, cols):
        return mcm.get_lat_lon_of_point_on_map(
            rows, cols,
            self.r_size, self.c_size,
            self.lat00, self.lon00,
            self.lat01, self.lon01,
            self.lat10, self.lon10,
            self.lat11, self.lon11,
            deg=True
        )

    def get_point_number_from_lattice_position(self, r, c):
        assert type(r) is int
        assert type(c) is int
        assert 0 <= r < self.r_size
        assert 0 <= c < self.c_size
        return r*self.r_size + c

    def get_lattice_position_from_point_number(self, p_i):
        assert type(p_i) is int
        assert 0 <= p_i < self.n_points
        r, c = divmod(p_i, self.c_size)
        assert 0 <= r < self.r_size
        assert 0 <= c < self.c_size
        return r, c

    def get_point_from_point_number(self, p_i):
        r, c = self.get_lattice_position_from_point_number(p_i)
        lat, lon = self.single_rc_to_latlon(r, c)
        xyz = mcm.unit_vector_lat_lon_to_cartesian(lat, lon, deg=True)
        return UnitSpherePoint({"latlondeg": (lat, lon), "xyz": xyz})
    
    def create_point_dicts(self):
        raise Exception("do not use anymore; memory-intensive")
        print("creating point dict for LatitudeLongitudeLattice")
        # creates dict to look up coordinates of point, e.g. {(x, y): UnitSpherePoint(...)}
        self.lattice_position_to_point_number = {}
        self.xyz_to_point_number = {}
        self.points = []

        # try to numpify, apply to whole array at once
        latlons_array = self.get_latlons_array()

        # latlons_array should first have two elements, for lat and lon
        assert latlons_array.shape[0] == 2
        # beyond that, expect a rectangular (2d) array of points, so the whole array has rank 3
        assert latlons_array.ndim == 3
        point_array_shape = latlons_array.shape[1:]
        assert point_array_shape == (self.c_size, self.r_size), "expected point array shape ({}, {}), got {}".format(self.r_size, self.c_size, point_array_shape)
        lats = latlons_array[0]
        lons = latlons_array[1]
        xyzs_array = mcm.unit_vector_lat_lon_to_cartesian(lats, lons)
        # now iterate over indices to create the individual point objects
        print("creating UnitSpherePoints")
        point_number = 0
        for x_i in range(self.r_size):
            if x_i % 100 == 0:
                print("progress: row {}/{}".format(x_i, self.r_size))
            for y_i in range(self.c_size):
                # x and y are indices in the point array
                lat = latlons_array[0, y_i, x_i]
                lon = latlons_array[1, y_i, x_i]
                latlon = (lat, lon)
                # also get xyz (cartesian in 3d, not to be confused with x and y on the rectangular lattice)
                x_coord = xyzs_array[0, y_i, x_i]
                y_coord = xyzs_array[1, y_i, x_i]
                z_coord = xyzs_array[2, y_i, x_i]
                xyz = (x_coord, y_coord, z_coord)
                coords_dict = {"xyz": xyz, "latlondeg": latlon}
                p = UnitSpherePoint(coords_dict)
                self.lattice_position_to_point_number[(x_i, y_i)] = point_number
                self.xyz_to_point_number[xyz] = point_number
                assert len(self.points) == point_number
                self.points.append(p)
                point_number += 1

        print("- done creating point dict")

    def get_usp_from_lattice_position(self, xy):
        point_number = self.lattice_position_to_point_number[xy]
        usp = self.points[point_number]
        return usp

    def get_point_number_from_usp(self, usp):
        return self.points.index(usp)

    def get_position_mathematical(self, point_number):
        return self.points[point_number].tuples

    def get_adjacencies(self):
        raise Exception("do not use anymore; memory-intensive")
        print("getting adjacencies for LatitudeLongitudeLattice")
        # build it from x, y first and then convert to UnitSpherePoint using the point_dict
        d_point_number = {}
        for x in range(self.r_size):
            for y in range(self.c_size):
                # 4 neighbors, not 8 (8 causes problems because rivers can flow through each other, for example)
                point_number = self.lattice_position_to_point_number[(x,y)]
                neighbors = [
                    (x+1, y), (x-1, y),
                    (x, y+1), (x, y-1),
                ]
                neighbors = self.filter_invalid_points(neighbors)
                neighbors_pn = [self.lattice_position_to_point_number[(x,y)] for (x,y) in neighbors]
                d_point_number[point_number] = neighbors_pn

        # convert to UnitSpherePoint
        # horribly inefficient, just do adjacencies by point number like you started to in icosa lattice, and if you need usp either calculate it or look it up from self.points
        #d_usp = {}
        #for k, neighbors_list in d.items():
        #    k_usp = self.get_usp_from_lattice_position(k)
        #    ns_usp = [self.get_usp_from_lattice_position(n) for n in neighbors_list]
        #    d_usp[k_usp] = ns_usp

        print("- done getting adjacencies")
        return d_point_number

    def average_latlon(self):
        half_x = self.r_size/2
        half_y = self.c_size/2
        return mcm.get_lat_lon_of_point_on_map(half_x, half_y, self.r_size, self.c_size,
            self.lat00, self.lon00,
            self.lat01, self.lon01,
            self.lat10, self.lon10,
            self.lat11, self.lon11,
            deg=True
        )

    def get_all_points_rc(self):
        return [(x, y) for x in range(self.r_size) for y in range(self.c_size)]

    def is_corner_pixel(self, x, y):
        return x in [0, self.r_size-1] and y in [0, self.c_size-1]

    def is_edge_pixel(self, x, y):
        x_edge = x in [0, self.r_size-1]
        y_edge = y in [0, self.c_size-1]
        return (x_edge and not y_edge) or (y_edge and not x_edge)

    def is_interior_pixel(self, x, y):
        return 1 <= x < self.r_size-1 and 1 <= y < self.c_size-1

    def get_representative_pixel(self, x, y):
        # for corner pixel, return itself since all 4 have different neighbor shape
        # for edge pixel, return one of the pixels on that edge
        # for interior pixel, return (1, 1)
        # then memoize only a total of 9 neighbor arrays for any size image, and just add offset
        x_edge = x in [0, self.r_size-1]
        y_edge = y in [0, self.c_size-1]
        if x_edge and y_edge:
            # corner
            return (x, y)
        elif x_edge or y_edge:
            # but not both, as that would have been caught by the previous condition
            # edge, replace the interior coordinate with 1
            if x_edge:
                return (x, 1)
            if y_edge:
                return (1, y)
        else:
            # interior
            return (1, 1)
    
    def size(self):
        return self.r_size * self.c_size

    def is_valid_point(self, x, y):  # flagged as slow due to sheer number of calls
        return 0 <= x < self.r_size and 0 <= y < self.c_size

    def filter_invalid_points(self, iterable):
        res = set()
        for p in iterable:
            if self.is_valid_point(*p):
                res.add(p)
        return res

    # def get_random_point(self, border_width=0):
    #     x = random.randrange(border_width, self.r_size - border_width)
    #     y = random.randrange(border_width, self.c_size - border_width)
    #     return (x, y)

    def get_next_step_in_path(self, current_point, objective, points_to_avoid):
        raise NotImplementedError
        # old: x and y grid only
        # # TODO this function can be adapted to the Lattice class, not relying on x and y
        # dx = objective[0] - current_point[0]
        # dy = objective[1] - current_point[1]
        # z = dx + dy*1j
        # objective_angle = np.angle(z, deg=True)
        # neighbors = self.get_neighbors(*current_point)
        # neighbors = [n for n in neighbors if n not in points_to_avoid]
        # neighbor_angles = [np.angle((n[0]-current_point[0]) + (n[1]-current_point[1])*1j, deg=True) for n in neighbors]
        # # set objective angle to zero for purposes of determining weight
        # neighbor_effective_angles = [abs(a - objective_angle) % 360 for a in neighbor_angles]
        # neighbor_effective_angles = [a-360 if a>180 else a for a in neighbor_effective_angles]
        # neighbor_weights = [180-abs(a) for a in neighbor_effective_angles]
        # assert all(0 <= w <= 180 for w in neighbor_weights), "{}\n{}\n{}".format(neighbors, neighbor_effective_angles, neighbor_weights)
        # total_weight = sum(neighbor_weights)
        # norm_weights = [w / total_weight for w in neighbor_weights]
        # chosen_one_index = np.random.choice([x for x in range(len(neighbors))], p=norm_weights)
        # chosen_one = neighbors[chosen_one_index]
        # return chosen_one

    def contains_point_latlon(self, p):
        if type(p) is UnitSpherePoint:
            p_lat, p_lon = p.latlondeg()
        else:
            p_lat, p_lon = p_latlon
        p_xyz = mcm.unit_vector_lat_lon_to_cartesian(p_lat, p_lon, deg=True)
        return det_in_tri.point_is_in_quadrilateral(p_xyz, self.xyz00, self.xyz01, self.xyz10, self.xyz11)



