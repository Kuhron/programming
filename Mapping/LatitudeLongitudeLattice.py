import numpy as np
import random

from Lattice import Lattice
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm


class LatitudeLongitudeLattice(Lattice):
    # a rectangular lattice on a part of the globe
    # can be warped to fit the quadrilateral between any four points
    # these points are called p00, p01, p10, p11 (row-column nomenclature)
    # e.g. p00 = Seattle, p01 = NYC, p10 = San Diego, p11 = Miami

    def __init__(self, x_size, y_size, latlon00, latlon01, latlon10, latlon11):
        self.x_size = x_size
        self.y_size = y_size
        self.x_range = np.arange(self.x_size)
        self.y_range = np.arange(self.y_size)
        self.lat00, self.lon00 = latlon00
        self.lat01, self.lon01 = latlon01
        self.lat10, self.lon10 = latlon10
        self.lat11, self.lon11 = latlon11

        self.create_point_dict()
        self.adjacencies = self.get_adjacencies()
        self.graph = self.get_graph()
        self.points = self.get_points()
    
    def create_point_dict(self):
        print("creating point dict for LatitudeLongitudeLattice")
        # creates dict to look up coordinates of point, e.g. {(x, y): UnitSpherePoint(...)}
        self.point_dict = {}

        # try to numpify, apply to whole array at once
        rows, cols = np.meshgrid(range(self.x_size), range(self.y_size))
        latlons_array = mcm.get_lat_lon_of_point_on_map(
            rows, cols,
            self.x_size, self.y_size,
            self.lat00, self.lon00,
            self.lat01, self.lon01,
            self.lat10, self.lon10,
            self.lat11, self.lon11,
            deg=True
        )
        raise

        # iterative
        # for x in range(self.x_size):
        #     print("x = {} / {}".format(x, self.x_size))
        #     for y in range(self.y_size):
        #         latlon = mcm.get_lat_lon_of_point_on_map(
        #             x, y, 
        #             self.x_size, self.y_size, 
        #             self.lat00, self.lon00, 
        #             self.lat01, self.lon01, 
        #             self.lat10, self.lon10, 
        #             self.lat11, self.lon11,
        #             deg=True
        #         )
        #         p = UnitSpherePoint(latlon, coords_system="latlondeg")
        #         self.point_dict[(x, y)] = p
        print("- done creating point dict")

    def get_adjacencies(self):
        print("getting adjacencies for LatitudeLongitudeLattice")
        # build it from x, y first and then convert to UnitSpherePoint using the point_dict
        d = {}
        for x in range(self.x_size):
            for y in range(self.y_size):
                # 4 neighbors, not 8 (8 causes problems because rivers can flow through each other, for example)
                neighbors = [
                    (x+1, y), (x-1, y),
                    (x, y+1), (x, y-1),
                ]
                neighbors = self.filter_invalid_points(neighbors)
                d[(x, y)] = neighbors

        # convert to UnitSpherePoint
        d_usp = {}
        for k, neighbors_list in d.items():
            k_usp = self.point_dict[k]
            ns_usp = [self.point_dict[n] for n in neighbors_list]
            d_usp[k_usp] = ns_usp

        print("- done getting adjacencies")
        return d_usp

    def average_latlon(self):
        half_x = self.x_size/2
        half_y = self.y_size/2
        return mcm.get_lat_lon_of_point_on_map(half_x, half_y, self.x_size, self.y_size,
            self.lat00, self.lon00,
            self.lat01, self.lon01,
            self.lat10, self.lon10,
            self.lat11, self.lon11,
            deg=True
        )

    def get_all_points(self):
        return {(x, y) for x in range(self.x_size) for y in range(self.y_size)}

    def is_corner_pixel(self, x, y):
        return x in [0, self.x_size-1] and y in [0, self.y_size-1]

    def is_edge_pixel(self, x, y):
        x_edge = x in [0, self.x_size-1]
        y_edge = y in [0, self.y_size-1]
        return (x_edge and not y_edge) or (y_edge and not x_edge)

    def is_interior_pixel(self, x, y):
        return 1 <= x < self.x_size-1 and 1 <= y < self.y_size-1

    def get_representative_pixel(self, x, y):
        # for corner pixel, return itself since all 4 have different neighbor shape
        # for edge pixel, return one of the pixels on that edge
        # for interior pixel, return (1, 1)
        # then memoize only a total of 9 neighbor arrays for any size image, and just add offset
        x_edge = x in [0, self.x_size-1]
        y_edge = y in [0, self.y_size-1]
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
        return self.x_size * self.y_size

    def is_valid_point(self, x, y):  # flagged as slow due to sheer number of calls
        return 0 <= x < self.x_size and 0 <= y < self.y_size

    def filter_invalid_points(self, iterable):
        res = set()
        for p in iterable:
            if self.is_valid_point(*p):
                res.add(p)
        return res

    def get_random_point(self, border_width=0):
        x = random.randrange(border_width, self.x_size - border_width)
        y = random.randrange(border_width, self.y_size - border_width)
        return (x, y)

