from UnitSpherePoint import UnitSpherePoint
import numpy as np
import random



class GreatCircleDistanceMatrix:
    # radius is the radius of the planet around which we are calculating these distances
    # too memory-intensive for large datasets to keep distances memorized, so just focus on calculating them efficiently on the fly every time
    def __init__(self, xyzs, radius=1):
        self.xyzs = xyzs
        n_points, three = xyzs.shape
        assert three == 3, xyzs.shape
        self.n_points = n_points
        self.radius = radius

    def __getitem__(self, index):
        a, b = index
        return GreatCircleDistanceMatrix.get_distance_array(a, b, self.radius)

    @staticmethod
    def get_distance_array(p, ps, radius=1):
        assert p.shape == (3,)
        n_points, three = ps.shape
        assert three == 3, ps.shape
        dx2 = (p-ps) ** 2
        d2 = sum(dx2.T)
        d_3d_unit_radius = d2 ** 0.5
        d_3d = radius * d_3d_unit_radius
        # now convert to great circle
        d_gc = UnitSpherePoint.convert_distance_3d_to_great_circle_array(d_3d, radius=radius)
        return d_gc

    @staticmethod
    def get_distance_dict(p, ps, radius=1):
        d_gc = GreatCircleDistanceMatrix.get_distance_array(p, ps, radius)
        res = {}
        for p2, dist in zip(ps, list(d_gc)):
            p2_tup = tuple(p2)
            res[p2_tup] = dist
        return res

    def get_distances_to_point(self, p):
        # p can be anything, not necessarily in the known points, but has to be on the unit sphere
        assert np.isclose(np.linalg.norm(p), 1), p
        ps = self.xyzs
        return GreatCircleDistanceMatrix.get_distance_dict(p, ps, self.radius)

    def get_points_within_distance_of_point(self, p, distance):
        distances = self.get_distances_to_point(p)
        return {xyz: d for xyz, d in distances.items() if d <= distance}


def get_nan_array(shape):
    a = np.empty(shape)
    a.fill(np.nan)
    return a


if __name__ == "__main__":
    xyzs = np.array([UnitSpherePoint.get_random_unit_sphere_point().xyz() for i in range(100)])
    matrix = GreatCircleDistanceMatrix(xyzs)
    for i in range(100):
        p_i = random.randrange(100)
        p = xyzs[p_i]
        p2_i = random.randrange(100)
        p2 = xyzs[p2_i]
        print("distance from", p, "to", p2, "is", matrix[p, p2])
