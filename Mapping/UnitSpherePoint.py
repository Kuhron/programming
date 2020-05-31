import numpy as np

import MapCoordinateMath as mcm


class UnitSpherePoint:
    def __init__(self, coords_tuple, coords_system):
        self.tuples = {
            "xyz": None,
            "latlondeg": None,
        }
        if coords_system == "xyz":
            self.tuples["xyz"] = coords_tuple
            self.tuples["latlondeg"] = mcm.unit_vector_cartesian_to_lat_lon(*coords_tuple, deg=True)
            check = mcm.unit_vector_lat_lon_to_cartesian(*self.tuples["latlondeg"], deg=True)
            diff = np.array(check) - np.array(coords_tuple)
            if np.linalg.norm(diff) > 1e-6:
                print("bad conversion:\ncoords_tuple: {}\ncheck: {}\ntuples: {}".format(coords_tuple, check, self.tuples))
        elif coords_system == "latlondeg":
            self.tuples["latlondeg"] = coords_tuple
            self.tuples["xyz"] = mcm.unit_vector_lat_lon_to_cartesian(*coords_tuple, deg=True)
            check = mcm.unit_vector_cartesian_to_lat_lon(*self.tuples["xyz"], deg=True)
            diff = np.array(check) - np.array(coords_tuple)
            if np.linalg.norm(diff) > 1e-6:
                print("bad conversion:\ncoords_tuple: {}\ncheck: {}\ntuples: {}".format(coords_tuple, check, self.tuples))
        else:
            raise ValueError("unrecognized coordinate system: {}".format(coords_system))

        self.point_data = {}
    
    def get_coords(self, coords_system):
        return self.tuples[coords_system]
    
    def distance(self, other):
        assert type(other) is UnitSpherePoint
        v0 = self.tuples["xyz"]
        v1 = other.tuples["xyz"]
        dv = np.array(v1) - np.array(v0)
        return np.linalg.norm(dv)
    
    def set_data(self, key, value):
        # use for giving the point elevation, rainfall, etc.
        # and will make it easier to transfer data to another point, e.g. when snapping to lattice
        self.point_data[key] = value
    
