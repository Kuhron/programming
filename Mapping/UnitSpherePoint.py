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
    def distance_3d_xyz_static(xyz1, xyz2, radius=1):
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        d = (dx**2 + dy**2 + dz**2) ** 0.5
        return d * radius

    @staticmethod
    def distance_3d_xyzs_to_xyz_static(xyzs, xyz, radius=1):
        three, n_points = xyzs.shape
        assert three == 3, f"xyzs shape should be (3, n) but got {xyzs.shape}"
        assert xyz.shape == (3,), f"xyz shape should be (3,) but got {xyz.shape}"
        diffs = xyzs - xyz
        assert diffs.shape == (3, n_points)
        diff2 = diffs ** 2
        diff2_sum = np.sum(diff2, axis=1)
        assert diff2_sum.shape == (n_points,)
        d = np.sqrt(diff2_sum)
        return d

    @staticmethod
    def distance_3d_latlondeg_static(latlon1, latlon2, radius=1):
        xyz1 = mcm.unit_vector_lat_lon_to_cartesian(*latlon1, deg=True)
        xyz2 = mcm.unit_vector_lat_lon_to_cartesian(*latlon2, deg=True)
        return UnitSpherePoint.distance_3d_xyz_static(xyz1, xyz2, radius=radius)

    @staticmethod
    def distance_great_circle_latlondeg_static(latlon1, latlon2, radius=1):
        d0 = UnitSpherePoint.distance_3d_latlondeg_static(latlon1, latlon2, radius=1)
        return UnitSpherePoint.convert_distance_3d_to_great_circle_single_value(d0, radius=radius)
        # don't multiply by radius twice, just do it in the great circle conversion call

    @staticmethod
    def distance_great_circle_xyz_static(xyz1, xyz2, radius=1):
        d0 = UnitSpherePoint.distance_3d_xyz_static(xyz1, xyz2, radius=1)
        return UnitSpherePoint.convert_distance_3d_to_great_circle_single_value(d0, radius=radius)
        # don't multiply by radius twice, just do it in the great circle conversion call

    @staticmethod
    def convert_distance_3d_to_great_circle_single_value(d0, radius=1):
        arr = np.array([d0])
        return UnitSpherePoint.convert_distance_3d_to_great_circle_array(arr, radius=radius)

    @staticmethod
    def convert_distance_3d_to_great_circle_array(d0, radius=1):
        r = radius
        theta = 2 * np.arcsin(d0 / (2*r))
        d_gc = r * theta
        # assert (0 <= d_gc).all(), f"bad great circle distance {d_gc} from d0={d0}, r={r}"
        # assert (d_gc <= np.pi * r).all(), f"bad great circle distance {d_gc} from d0={d0}, r={r}"
        # assert ((d_gc > d0) | (abs(d_gc - d0) < 1e-9)).all(), f"shortest distance should be a straight line, but got great-circle {d_gc} from Euclidean {d0}"
        # print(f"d0 = {d0}, r = {r} -> great circle distance {d_gc}")
        return d_gc
        # return (np.vectorize(lambda d: UnitSpherePoint.convert_distance_3d_to_great_circle_single_value(d, radius=radius)))(d0)
    
    @staticmethod
    def convert_distance_great_circle_to_3d(d_gc, radius=1):
        r = radius
        theta = d_gc / r
        d0 = 2 * r * np.sin(theta)
        assert 0 <= d0 <= 2*r, f"bad 3d distance {d0} from d_gc={d_gc}, r={r}"
        assert d0 <= d_gc, "shortest distance should be a straight line"
        return d0

    @staticmethod
    def convert_distance_great_circle_to_3d_array(d_gc, radius=1):
        # TODO if ever use this much, convert it to working primarily on arrays rather than single values
        # (like you did with the conversion of 3d to great circle distance)
        return (np.vectorize(lambda d: UnitSpherePoint.convert_distance_great_circle_to_3d(d, radius=radius)))(d_gc)

    def distance_3d(self, other, radius=1):
        assert type(other) is UnitSpherePoint
        v0 = self.tuples["xyz"]
        v1 = other.tuples["xyz"]
        dv = np.array(v1) - np.array(v0)
        d = np.linalg.norm(dv)
        return d * radius

    def distance_great_circle(self, other, radius):
        d0 = self.distance_3d(other, radius=1)
        return UnitSpherePoint.convert_distance_3d_to_great_circle_single_value(d0, radius=radius)
        # multiply by radius once only

    def set_data(self, key, value):
        # use for giving the point elevation, rainfall, etc.
        # and will make it easier to transfer data to another point, e.g. when snapping to lattice
        self.point_data[key] = value
    
    @staticmethod
    def get_midpoint(p0, p1):
        xyz0 = p0.get_coords("xyz")
        xyz1 = p1.get_coords("xyz")
        midpoint_normalized_xyz = mcm.get_unit_sphere_midpoint_from_xyz(xyz0, xyz1)
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
    def from_xyz(x, y, z, point_number=None):
        xyz = np.array([x, y, z])
        latlondeg = mcm.unit_vector_cartesian_to_lat_lon(x, y, z, deg=True)
        return UnitSpherePoint({"xyz":xyz, "latlondeg":latlondeg}, point_number=point_number)

    @staticmethod
    def from_latlondeg(lat, lon, point_number=None):
        latlondeg = np.array([lat, lon])
        xyz = mcm.unit_vector_lat_lon_to_cartesian(lat, lon, deg=True)
        return UnitSpherePoint({"xyz":xyz, "latlondeg":latlondeg}, point_number=point_number)

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

