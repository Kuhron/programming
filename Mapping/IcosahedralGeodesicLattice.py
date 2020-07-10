import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree

from Lattice import Lattice
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm


class IcosahedralGeodesicLattice(Lattice):
    EARTH_RADIUS_KM = 6371
    CADA_II_RADIUS_FACTOR = 2.116
    CADA_II_RADIUS_KM = CADA_II_RADIUS_FACTOR * EARTH_RADIUS_KM

    def __init__(self, edge_length_km=None, iterations=None):
        assert int(edge_length_km is None) + int(iterations is None) == 1, "need either edge_length_km or iterations, not both, got {} and {}".format(edge_length_km, iterations)
        self.edge_length_km = edge_length_km
        if iterations is not None:
            assert iterations % 1 == 0, "need int value for iterations if supplied, got {}".format(iterations)
            iterations = int(iterations)
        self.iterations = iterations
        ordered_points, adjacencies_by_point_index = self.get_adjacencies()
        self.adjacencies_by_point_index = adjacencies_by_point_index
        self.points = ordered_points
        self.usp_to_index = self.get_usp_to_index()
        self.adjacencies = self.convert_adjacencies_to_usp()
        self.xyz_coords = []
        self.xyz_to_point_number = {}
        for point_number, p in enumerate(self.points):
            xyz = tuple(p.get_coords("xyz"))
            self.xyz_coords.append(xyz)
            self.xyz_to_point_number[xyz] = point_number
        self.xyz_coords = np.array(self.xyz_coords)
        self.kdtree = KDTree(self.xyz_coords)
        self.graph = self.get_graph()

    def get_adjacencies(self):
        cada_ii_radius_km = IcosahedralGeodesicLattice.CADA_II_RADIUS_KM
        if self.iterations is not None:
            iterations_needed = self.iterations
        elif self.edge_length_km is not None:
            # edge_length_km determines how high the resolution is
            # derive from the inverse formula at https://en.wikipedia.org/wiki/Regular_icosahedron
            # radius of sphere that touches icosa at all vertices = (edge_length)/4 * sqrt(10 + 2*sqrt(5))
            icosa_edge_length_from_radius_to_vertex = lambda r: r * 4 / np.sqrt(10 + 2 * np.sqrt(5))
            initial_edge_length_km = icosa_edge_length_from_radius_to_vertex(cada_ii_radius_km)
            factor = initial_edge_length_km / self.edge_length_km
            # each iteration halves the edge length
            iterations_needed = int(np.ceil(np.log2(factor)))
        else:
            raise

        if iterations_needed > 7:
            print("You requested {} iterations of precision IcosahedralGeodesicLattice, but it is memory-intensive to go above 7.".format(iterations_needed))
            input("press enter to continue if desired")

        try:
            return IcosahedralGeodesicLattice.get_adjacencies_from_memoization_file(iterations_needed)
        except FileNotFoundError:
            print("retrieving memoized icosahedron files for {} iterations failed; constructing from scratch".format(iterations_needed))
        

        icosahedron_original_points_latlon = {
            # north pole
            "NP": (90, 0),
            # north ring of five points, star with a point at lon 0
            "NR0": (30, 0), "NRp72": (30, 72), "NRp144": (30, 144), "NRm72": (30, -72), "NRm144": (30, -144),
            # south ring of five points, star with a point at lon 180
            "SR180": (-30, 180), "SRp108": (-30, 108), "SRp36": (-30, 36), "SRm108": (-30, -108), "SRm36": (-30, -36),
            # south pole
            "SP": (-90, 0),
        }
        original_points_neighbors_by_name = {
            # start with "north" neighbor, i.e., directly left on peel-rectangle representation (for poles the order should still obey counterclockwise, but starting point doesn't matter)
            # poles
            "NP": ["NR0", "NRp72", "NRp144", "NRm144", "NRm72"],  # going eastward (counterclockwise)
            "SP": ["SR180", "SRp108", "SRp36", "SRm36", "SRm108"],  # going westward (counterclockwise)

            # peel 0
            "NR0": ["NP", "NRm72", "SRm36", "SRp36", "NRp72"],
            "SRp36": ["NR0", "SRm36", "SP", "SRp108", "NRp72"],

            # peel 1
            "NRp72": ["NP", "NR0", "SRp36", "SRp108", "NRp144"],
            "SRp108": ["NRp72", "SRp36", "SP", "SR180", "NRp144"],

            # peel 2
            "NRp144": ["NP", "NRp72", "SRp108", "SR180", "NRm144"],
            "SR180": ["NRp144", "SRp108", "SP", "SRm108", "NRm144"],

            # peel 3
            "NRm144": ["NP", "NRp144", "SR180", "SRm108", "NRm72"],
            "SRm108": ["NRm144", "SR180", "SP", "SRm36", "NRm72"],

            # peel 4
            "NRm72": ["NP", "NRm144", "SRm108", "SRm36", "NR0"],
            "SRm36": ["NRm72", "SRm108", "SP", "SRp36", "NR0"],
        }
        assert len(original_points_neighbors_by_name) == 12 and all(len(vals) == 5 for vals in original_points_neighbors_by_name.values())
        # check transitivity of neighborliness, since I input the lists manually
        for point_name, neighbors in original_points_neighbors_by_name.items():
            for neigh in neighbors:
                assert point_name in original_points_neighbors_by_name[neigh], "intransitive adjacency with {} and {}".format(point_name, neigh)
        # icosahedron points are correct now; time to divide edges

        # put them in this ordering convention:
        # north pole first, south pole second, omit these from all expansion operations, by only operating on points[2:] (non-pole points)
        # new points are appended to the point list in the order they are created
        # order the neighbors in the following order, and only bisect the edges from each point to the first three:
        # - [north, west, southwest, (others)]. order of others doesn't matter that much, can just keep going counterclockwise
        # ordering of neighbors for poles thus doesn't matter as that list will never be used for expansion

        original_points_order_by_name = [
            "NP", "SP",  # poles
            "NR0", "SRp36",  # peel 0
            "NRp72", "SRp108",  # peel 1
            "NRp144", "SR180",  # peel 2
            "NRm144", "SRm108",  # peel 3
            "NRm72", "SRm36",  # peel 4
        ]

        # keep the point objects in a single array that can be indexed by point index
        # the rest of the data, i.e., the adjacencies dictionary, should be all in terms of integer indices that refer to the points array

        ordered_points = []
        adjacencies_by_point_index = {}

        # place original points in the list
        for p_name in original_points_order_by_name:
            point_index = len(ordered_points)
            p_latlon = icosahedron_original_points_latlon[p_name]
            p_xyz = mcm.unit_vector_lat_lon_to_cartesian(*p_latlon)
            coords_dict = {"xyz": p_xyz, "latlondeg": p_latlon}
            usp = UnitSpherePoint(coords_dict)
            ordered_points.append(usp)
        assert len(ordered_points) == 12, "initial icosa needs 12 vertices"

        # add their neighbors by index
        for point_index in range(len(ordered_points)):
            point_name = original_points_order_by_name[point_index]
            neighbor_names = original_points_neighbors_by_name[point_name]
            neighbor_indices = [original_points_order_by_name.index(name) for name in neighbor_names]
            adjacencies_by_point_index[point_index] = neighbor_indices
            print("adjacencies now:\n{}\n".format(adjacencies_by_point_index))

        # bisect edges until reach resolution
        print("bisecting for {} iterations".format(iterations_needed))
        for iteration_i in range(1, iterations_needed+1):
        # while True:
            # # check some random edges to get average edge length
            # edge_lengths = []
            # for _ in range(100):
            #     random_point_index = random.choice(list(adjacencies_by_point_index.keys()))
            #     neighbor_index = random.choice(adjacencies_by_point_index[random_point_index])
            #     p0 = ordered_points[random_point_index]
            #     p1 = ordered_points[neighbor_index]
            #     angle_radians = UnitSpherePoint.get_angle_radians_between(p0, p1)
            #     edge_length = cada_ii_radius_km * angle_radians
            #     edge_lengths.append(edge_length)
            # edge_length = np.mean(edge_lengths)
            # print("edge_length = {} km, iteration {}".format(edge_length, iteration_i))
            # if edge_length <= self.edge_length_km:
            #     break
            # else:
            #     iteration_i += 1
            #     # we checked e.g. iteration 0 the first time (just the icosahedron's 12 vertices), now are going to enter new iteration

            point_index_this_iteration_started_at = len(ordered_points)
            last_iteration_i = iteration_i - 1
            expected_index = 2 + 10 * 2 ** (2 * last_iteration_i)
            assert point_index_this_iteration_started_at == expected_index, "math error in number of points by iteration at i={}, expected {}, got {}".format(iteration_i, expected_index, point_index_this_iteration_started_at)

            # bisection and neighbor updating

            # don't add any neighbors for the poles
            # for all other points (inside the 5 peels), bisect the first three edges (north(west)ward, westward, and south(west)ward, roughly)
            # thus each existing peel point will add three new points

            # call the directions (on rectangle representation) L, DL, D, R, UR, U (in counterclockwise order, as they appear on the rectangle representation for a generic peel-internal point)
            get_opposite_neighbor_direction = lambda i: {0: 3, 3: 0, 1: 4, 4: 1, 2: 5, 5: 2}[i]  # map L vs R, DL vs UR, D vs U

            peel_point_indices = range(2, len(ordered_points))
            for p_i in peel_point_indices:
                first_three_neighbors = adjacencies_by_point_index[p_i][:3]

                # bisect these edges
                for neighbor_direction_i, n_i in enumerate(first_three_neighbors):
                    p0 = ordered_points[p_i]
                    p1 = ordered_points[n_i]
                    midpoint = UnitSpherePoint.get_midpoint(p0, p1)

                    # add it to the list so it will be in order of creation
                    new_point_index = len(ordered_points)
                    ordered_points.append(midpoint)

                    adjacencies_by_point_index[new_point_index] = [None] * 6  # all new points will have 6 neighbors, only the 12 original vertices have 5
                    neighbor_direction_back_to_parent = get_opposite_neighbor_direction(neighbor_direction_i)

                    # pi_index_ni and ni_index_pi are the ACTUAL INDICES IN THE ADJACENCY LIST, which will DIFFER for the original 12 vertices because they only have 5 neighbors
                    # for interior (later-added) points, these will coincide with neighbor_direction_i and neighbor_direction_back_to_parent
                    # print("pi>ni = {} , ni>pi = {}".format(pi_index_ni, ni_index_pi))

                    # update the adjacencies by replacing the original neighbor with the midpoint
                    # print("pre-replacement adjacency of parent {} = {}".format(p_i, adjacencies_by_point_index[p_i]))
                    pi_index_ni = adjacencies_by_point_index[p_i].index(n_i)
                    adjacencies_by_point_index[p_i][pi_index_ni] = new_point_index
                    adjacencies_by_point_index[new_point_index][neighbor_direction_i] = n_i  # here use direction, not index, in case parent has 5 neighbors
                    # print("post-replacement adjacency of parent {} = {}".format(p_i, adjacencies_by_point_index[p_i]))

                    # the same must be done for the original neighbor as well
                    # print("pre-replacement adjacency of neighbor {} = {}".format(n_i, adjacencies_by_point_index[n_i]))
                    ni_index_pi = adjacencies_by_point_index[n_i].index(p_i)
                    adjacencies_by_point_index[n_i][ni_index_pi] = new_point_index
                    adjacencies_by_point_index[new_point_index][neighbor_direction_back_to_parent] = p_i  # here use direction, not index, in case parent has 5 neighbors
                    # print("post-replacement adjacency of neighbor {} = {}".format(n_i, adjacencies_by_point_index[n_i]))

                    # print("post-replacement adjacency of new point {} = {}".format(new_point_index, adjacencies_by_point_index[new_point_index]))

                # print("finished bisecting first three neighbors for point {}".format(p_i))

            print("finished bisecting edges for iteration {}".format(iteration_i))

            print("\nfilling in missing neighbors for new points this iteration (i={})\n".format(iteration_i))

            # flanking approach to filling in missing neighbors:
            # each new point will have six adjacencies total, two of them currently known, and those two will be directly opposite one another
            # e.g. we know that 0 borders 36, 12, and 18 (in that counterclockwise order), and we know where 0 is in the adjacency list of 12
            # - then the two which "flank" 12 in 0's list (i.e. 36 and 18) will be in the OPPOSITE order in 12's list, flanking 0.
            # this can be seen by drawing a rhombus from the two triangles:
            #  /- 18 -\
            # 0 ------ 12
            #  \- 36 -/

            indices_to_fill_out = list(range(point_index_this_iteration_started_at, len(ordered_points)))
            indices_filled_to_completion = list()

            for p_i in indices_to_fill_out:
                adjacencies_p = adjacencies_by_point_index[p_i]
                # print("filling out adjacencies for point {}: originally {}".format(p_i, adjacencies_p))
                assert len(adjacencies_p) == 6, "all new points should have 6 neighbors but got {}".format(adjacencies_p)
                assert sum(x is None for x in adjacencies_p) == 4, "expected 4 Nones and two known neighbors but got {}".format(adjacencies_p)
                # get the 2 known neighbors
                neighbor_0 = None
                n_i_0 = None
                neighbor_1 = None
                n_i_1 = None
                for n_i, neighbor in enumerate(adjacencies_p):
                    if neighbor is not None:
                        if n_i_0 is None:
                            assert neighbor_0 is None
                            n_i_0 = n_i
                            neighbor_0 = neighbor
                        else:
                            assert neighbor_0 is not None
                            assert n_i_1 is None
                            assert neighbor_1 is None
                            n_i_1 = n_i
                            neighbor_1 = neighbor

                for n_i, n in zip([n_i_0, n_i_1], [neighbor_0, neighbor_1]):
                    adjacencies_n = adjacencies_by_point_index[n]
                    index_p_in_n = adjacencies_n.index(p_i)
                    position_plus_from_n = (index_p_in_n + 1) % len(adjacencies_n)
                    position_minus_from_n = (index_p_in_n - 1) % len(adjacencies_n)
                    flank_plus_from_n = adjacencies_n[position_plus_from_n]
                    flank_minus_from_n = adjacencies_n[position_minus_from_n]
                    # place the flanks in adjacencies_p so that they flank n but in the opposite order
                    position_plus_from_p = (n_i + 1) % len(adjacencies_p)
                    position_minus_from_p = (n_i - 1) % len(adjacencies_p)
                    adjacencies_p[position_plus_from_p] = flank_minus_from_n  # OPPOSITE PLUS/MINUS
                    adjacencies_p[position_minus_from_p] = flank_plus_from_n  # OPPOSITE PLUS/MINUS
            
                # print("- got filled neighbors for {}: {}".format(p_i, adjacencies_p))
                assert adjacencies_by_point_index[p_i] == adjacencies_p  # checking that modifying object by reference is working
                indices_filled_to_completion.append(p_i)
            
            for p_i in indices_filled_to_completion:
                # don't do this in the loop or it'll skip stuff
                indices_to_fill_out.remove(p_i)

            print("- finished filling out adjacencies for new points")
            assert len(indices_to_fill_out) == 0, "leftover points did not get filled out: {}".format(indices_to_fill_out)
            for p_i in range(len(ordered_points)):
                assert all(x is not None for x in adjacencies_by_point_index[p_i]), "Nones left in adjacencies for {}: {}".format(p_i, adjacencies_by_point_index[p_i])

            position_memo_fp = "/home/wesley/programming/Mapping/MemoIcosa/MemoIcosaPosition_Iteration{}.txt".format(iteration_i)
            adjacency_memo_fp = "/home/wesley/programming/Mapping/MemoIcosa/MemoIcosaAdjacency_Iteration{}.txt".format(iteration_i)
            print("writing icosahedron memoization files")
            with open(position_memo_fp, "w") as f:
                for p_i in range(len(ordered_points)):
                    usp = ordered_points[p_i]
                    x, y, z = usp.get_coords("xyz")
                    lat, lon = usp.get_coords("latlondeg")
                    s = "{}:{},{},{};{},{}\n".format(p_i, x, y, z, lat, lon)
                    f.write(s)
            with open(adjacency_memo_fp, "w") as f:
                for p_i in range(len(ordered_points)):
                    neighbor_indices_str = ",".join(str(x) for x in adjacencies_by_point_index[p_i])
                    s = "{}:{}\n".format(p_i, neighbor_indices_str)
                    f.write(s)
            print("- done writing memoization files")

            print("now have {} points, iteration {}".format(len(ordered_points), iteration_i))

        return ordered_points, adjacencies_by_point_index

    @staticmethod
    def get_adjacencies_from_memoization_file(iteration):
        # still not using this, can do so if creation of icosa is slow but for now it's fine
        adjacencies_fp = "MemoIcosa/MemoIcosaAdjacency_Iteration{}.txt".format(iteration)
        positions_fp = "MemoIcosa/MemoIcosaPosition_Iteration{}.txt".format(iteration)
        adjacencies_by_point_index = {}
        ordered_points = []

        with open(adjacencies_fp) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            p_i, adjacencies_list = line.split(":")
            p_i = int(p_i)
            adjacencies = [int(x) for x in adjacencies_list.split(",")]
            adjacencies_by_point_index[p_i] = adjacencies

        with open(positions_fp) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            p_i, coords = line.split(":")
            xyz_coords, latlon_coords = coords.split(";")
            x, y, z = xyz_coords.split(",")
            lat, lon = latlon_coords.split(",")
            p_i = int(p_i)
            assert p_i == len(ordered_points), "out-of-order point numbering in {} at p_i={}".format(positions_fp, p_i)
            x = float(x)
            y = float(y)
            z = float(z)
            lat = float(lat)
            lon = float(lon)
            coords_dict = {"xyz": (x, y, z), "latlondeg": (lat, lon)}
            usp = UnitSpherePoint(coords_dict)
            ordered_points.append(usp)

        print("successfully retrieved icosahedron memoization for {} iterations".format(iteration))
        return ordered_points, adjacencies_by_point_index

    def convert_adjacencies_to_usp(self):
        adjacencies_usp = {}
        for k, lst in self.adjacencies_by_point_index.items():
            usp = self.points[k]
            adj_usps = [self.points[n] for n in lst]
            adjacencies_usp[usp] = adj_usps
        return adjacencies_usp

    def get_usp_to_index(self):
        d = {}
        for i in range(len(self.points)):
            d[self.points[i]] = i
        return d

    @staticmethod
    def get_iterations_from_number_of_points(n):
        try:
            return {12: 0, 42: 1, 162: 2, 642: 3, 2562: 4, 10242: 5}[n]
        except KeyError:
            power2 = (n - 2) / 10
            assert power2 % 1 == 0, "invalid number of points {}, should be equal to power of 2 times 10 plus 2, e.g. 642 or 10242".format(n)
            log2 = np.log2(power2)
            assert log2 % 1 == 0, "invalid power of 2: {}".format(power2)
            two_i = int(log2)
            iterations = two_i / 2
            assert iterations % 1 == 0, "invalid 2i: {}".format(two_i)
            return iterations

    def get_next_step_in_path(self, current_point, objective, points_to_avoid):
        # get vector to objective, project it onto plane tangent to sphere at current_point
        # vector rejection of a from b is the component of a that is leftover when you subtract the projection of a onto b
        # this is what I want (b from core to current_point or vice versa, a from current_point to objective, rejection tells what direction to go in from a)
        a_i, b_i = current_point, objective
        xyz_a = np.array(self.points[a_i].get_coords("xyz"))
        xyz_b = np.array(self.points[b_i].get_coords("xyz"))
        vector_to_objective = xyz_b - xyz_a
        vector_from_core_to_current = xyz_a  # minus (0, 0, 0)
        rejection_vector = mcm.vector_rejection_3d(vector_to_objective, vector_from_core_to_current)
        # now get neighbor in that direction, but allow some randomness somehow
        neighbor_indices = self.adjacencies_by_point_index[current_point]
        neighbor_xyzs = [np.array(self.points[p_i].get_coords("xyz")) for p_i in neighbor_indices]
        displacements = [xyz - xyz_a for xyz in neighbor_xyzs]
        angles = [mcm.angle_between_vectors(dis, rejection_vector) for dis in displacements]
        # bigger weight for smaller angle (closer to target direction), biggest angle will be pi
        weights = [np.pi - angle for angle in angles]
        weights = np.array(weights) / sum(weights)
        chosen_neighbor_point_index = np.random.choice(neighbor_indices, p=weights)
        return chosen_neighbor_point_index



if __name__ == "__main__":
    edge_length_km = 250
    test_lattice = IcosahedralGeodesicLattice(edge_length_km)
    # test_lattice.plot_points()
    data = test_lattice.place_random_data()
    test_lattice.plot_data(data)
    
