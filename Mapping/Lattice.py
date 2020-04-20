# base class to inherit from for different lattice types
# e.g. lat/lon or geodesic
# this class should implement the methods that exist regardless of the grid topology
# e.g. finding nearest neighbors, finding all neighbors, converting to xyz, etc.

from UnitSpherePoint import UnitSpherePoint


class Lattice:
    def __init__(self):
        raise NotImplementedError("do not initialize Lattice itself; use a subclass such as LatitudeLongitudeLattice")

    def get_adjacencies(self):
        # specific to the subclasses, depending on type of lattice
        raise NotImplementedError

    def get_nearest_lattice_point_to_input_point(p):
        assert type(p) is UnitSpherePoint
        # first pass: brute force
        return min(self.adjacencies, key=lambda x: x.distance(p))

        # later could optimize somehow, e.g. take only a box of +/- dx,dy,dz and sort those

