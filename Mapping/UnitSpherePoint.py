import numpy as np

import MapCoordinateMath as mcm


class UnitSpherePoint:
    def __init__(self, coords_dict, point_number=None):
        self.tuples = {
            "xyz": None,
            "latlondeg": None,
        }
        self.point_number = point_number

        for coords_system, coords_tuple in coords_dict.items():
            if type(coords_tuple) is not tuple:
                coords_tuple = tuple(coords_tuple)
            if coords_system == "xyz":
                x,y,z = coords_tuple  # catch shape problems
                self.tuples["xyz"] = coords_tuple
               #self.tuples["latlondeg"] = mcm.unit_vector_cartesian_to_lat_lon(*coords_tuple, deg=True)
                #check = mcm.unit_vector_lat_lon_to_cartesian(*self.tuples["latlondeg"], deg=True)
                #diff = np.array(check) - np.array(coords_tuple)
                #if np.linalg.norm(diff) > 1e-6:
                #    print("bad conversion:\ncoords_tuple: {}\ncheck: {}\ntuples: {}".format(coords_tuple, check, self.tuples))
            elif coords_system == "latlondeg":
                lat,lon = coords_tuple  # catch shape problems
                self.tuples["latlondeg"] = coords_tuple
                #self.tuples["xyz"] = mcm.unit_vector_lat_lon_to_cartesian(*coords_tuple, deg=True)
                #check = mcm.unit_vector_cartesian_to_lat_lon(*self.tuples["xyz"], deg=True)
                #diff = np.array(check) - np.array(coords_tuple)
                #if np.linalg.norm(diff) > 1e-6:
                #    print("bad conversion:\ncoords_tuple: {}\ncheck: {}\ntuples: {}".format(coords_tuple, check, self.tuples))
            else:
                raise ValueError("unrecognized coordinate system: {}".format(coords_system))

        # require the coords to all be specified from now on, don't calculate them here because you will not be able to take advantage of parallel array computing if you re-run the conversion function for every time this class is instantiated
        if any(x is None for x in self.tuples.values()):
            raise ValueError("passing both xyz and latlondeg is required, but got {}".format(self.tuples))

        self.point_data = {}

    def __repr__(self):
        x, y, z = self.get_coords("xyz")
        lat, lon = self.get_coords("latlondeg")
        return "<USP #{} (x={:+}, y={:+}, z={:+}) (lat={:+} deg, lon={:+} deg)>".format(self.point_number, x, y, z, lat, lon)
    
    def get_coords(self, coords_system):
        return self.tuples[coords_system]

    def xyz(self, as_array=False):
        xyz = self.tuples["xyz"]
        if as_array:
            return np.array(xyz)
        else:
            return xyz

    def latlondeg(self, as_array=False):
        latlon = self.tuples["latlondeg"]
        if as_array:
            return np.array(latlon)
        else:
            return latlon

    def latlonrad(self):
        tup = self.latlondeg()
        return tuple(x*np.pi/180 for x in tup)

    def latlon(self):
        raise Exception("please use .latlondeg() or .latlonrad()")

    @staticmethod
    def distance_3d_latlondeg_static(latlon1, latlon2):
        xyz1 = mcm.unit_vector_lat_lon_to_cartesian(*latlon1, deg=True)
        xyz2 = mcm.unit_vector_lat_lon_to_cartesian(*latlon2, deg=True)
        dv = np.array(xyz1) - np.array(xyz2)
        return np.linalg.norm(dv)

    @staticmethod
    def distance_great_circle_latlondeg_static(latlon1, latlon2):
        d0 = UnitSpherePoint.distance_3d_latlondeg_static(latlon1, latlon2)
        r = 1
        return UnitSpherePoint.convert_distance_3d_to_great_circle(d0, r)

    @staticmethod
    def convert_distance_3d_to_great_circle(d0, r):
        theta = 2 * np.arcsin(d0 / (2*r))
        d_gc = r * theta
        assert 0 <= d_gc <= np.pi * r
        assert d_gc >= d0, "shortest distance should be a straight line"
        return d_gc

    def distance_3d(self, other):
        assert type(other) is UnitSpherePoint
        v0 = self.tuples["xyz"]
        v1 = other.tuples["xyz"]
        dv = np.array(v1) - np.array(v0)
        return np.linalg.norm(dv)

    def distance_great_circle(self, other):
        d0 = self.distance_3d(other)
        r = 1
        return UnitSpherePoint.convert_distance_3d_to_great_circle(d0, r)

    def set_data(self, key, value):
        # use for giving the point elevation, rainfall, etc.
        # and will make it easier to transfer data to another point, e.g. when snapping to lattice
        self.point_data[key] = value
    
    @staticmethod
    def get_midpoint(p0, p1):
        xyz0 = p0.get_coords("xyz")
        xyz1 = p1.get_coords("xyz")
        midpoint_raw_xyz = (np.array(xyz0) + np.array(xyz1)) / 2
        midpoint_normalized_xyz = tuple((midpoint_raw_xyz / np.linalg.norm(midpoint_raw_xyz)).reshape(3))
        midpoint_normalized_latlon = mcm.unit_vector_cartesian_to_lat_lon(*midpoint_normalized_xyz, deg=True)
        coords_dict = {"xyz": midpoint_normalized_xyz, "latlondeg": midpoint_normalized_latlon}
        return UnitSpherePoint(coords_dict)

    @staticmethod
    def get_angle_radians_between(p0, p1):
        xyz0 = np.array(p0.get_coords("xyz"))
        xyz1 = np.array(p1.get_coords("xyz"))
        return mcm.angle_between_vectors(xyz0, xyz1)

    def get_immutable(self):
        tuple_keys = sorted(self.tuples.keys())
        lst = []
        for k in tuple_keys:
            tup_piece = (k, self.tuples[k])
            lst.append(tup_piece)
        return tuple(lst)

    def __hash__(self):
        # raise Exception("Warning: USP hashing is not yet reliable, still get different immutable objects with same coordinates, possibly due to rounding errors. Please revise code to use point index or something else that can reliably point to the same USP object.")
        raise Exception("this is slow; remove as many calls to indexing on USP as possible")
        return hash(self.get_immutable())

    @staticmethod
    def get_random_unit_sphere_point():
        a = np.random.normal(0,1,(3,))
        a /= np.linalg.norm(a)
        xyz = a
        return UnitSpherePoint.from_xyz(*xyz)

    @staticmethod
    def from_xyz(x, y, z):
        xyz = np.array([x, y, z])
        latlondeg = mcm.unit_vector_cartesian_to_lat_lon(x, y, z, deg=True)
        return UnitSpherePoint({"xyz":xyz, "latlondeg":latlondeg})

    @staticmethod
    def from_latlondeg(lat, lon):
        latlondeg = np.array([lat, lon])
        xyz = mcm.unit_vector_lat_lon_to_cartesian(lat, lon, deg=True)
        return UnitSpherePoint({"xyz":xyz, "latlondeg":latlondeg})

    @staticmethod
    def random():
        return UnitSpherePoint.get_random_unit_sphere_point()  # alias

    @staticmethod
    def random_within_latlon_box(n_points, min_lat, max_lat, min_lon, max_lon):
        res = []
        while len(res) < n_points:
            usp = UnitSpherePoint.random()
            lat, lon = usp.latlondeg()
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                res.append(usp)
        return res

