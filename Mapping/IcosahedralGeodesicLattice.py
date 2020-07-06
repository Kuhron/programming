import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.spatial import KDTree

from Lattice import Lattice
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm


class IcosahedralGeodesicLattice(Lattice):
    EARTH_RADIUS_KM = 6371
    CADA_II_RADIUS_FACTOR = 2.116
    CADA_II_RADIUS_KM = CADA_II_RADIUS_FACTOR * EARTH_RADIUS_KM

    def __init__(self, edge_length_km):
        self.edge_length_km = edge_length_km
        self.adjacencies = self.get_adjacencies()
        self.points = list(self.adjacencies.keys())
        self.xyz_coords = []
        self.xyz_to_point_number = {}
        for point_number, p in enumerate(self.points):
            xyz = p.get_coords("xyz")
            self.xyz_coords.append(xyz)
            self.xyz_to_point_number[xyz] = point_number
        self.kdtree = KDTree(self.xyz_coords)
        self.graph = self.get_graph()

    def get_adjacencies(self):
        # edge_length_km determines how high the resolution is
        cada_ii_radius_km = IcosahedralGeodesicLattice.CADA_II_RADIUS_KM

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

        # add their neighbors by index
        for point_index in range(len(ordered_points)):
            point_name = original_points_order_by_name[point_index]
            neighbor_names = original_points_neighbors_by_name[point_name]
            neighbor_indices = [original_points_order_by_name.index(name) for name in neighbor_names]
            adjacencies_by_point_index[point_index] = neighbor_indices
            print("adjacencies now:\n{}\n".format(adjacencies_by_point_index))

        # bisect edges until reach resolution
        iteration_i = 0
        while True:
            # check some random edges to get average edge length
            edge_lengths = []
            for _ in range(100):
                random_point_index = random.choice(list(adjacencies_by_point_index.keys()))
                neighbor_index = random.choice(adjacencies_by_point_index[random_point_index])
                # v0 = mcm.unit_vector_lat_lon_to_cartesian(*random_point)
                # v1 = mcm.unit_vector_lat_lon_to_cartesian(*neigh)
                p0 = ordered_points[random_point_index]
                p1 = ordered_points[neighbor_index]
                angle_radians = UnitSpherePoint.get_angle_radians_between(p0, p1)
                edge_length = cada_ii_radius_km * angle_radians
                edge_lengths.append(edge_length)
            edge_length = np.mean(edge_lengths)
            print("edge_length = {} km, iteration {}".format(edge_length, iteration_i))
            if edge_length <= self.edge_length_km:
                break
            else:
                iteration_i += 1
                # we checked e.g. iteration 0 the first time (just the icosahedron's 12 vertices), now are going to enter new iteration

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
            l_new_point_indices = []
            dl_new_point_indices = []
            d_new_point_indices = []

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

                    # no longer used; from dldl approach, which failed, but keep this code until have an approach that does work
                    # if neighbor_direction_i == 0:
                    #     l_new_point_indices.append(new_point_index)
                    # elif neighbor_direction_i == 1:
                    #     dl_new_point_indices.append(new_point_index)
                    # elif neighbor_direction_i == 2:
                    #     d_new_point_indices.append(new_point_index)
                    # else:
                    #     raise ValueError("neighbor direction should only be 0, 1, or 2, got {}".format(neighbor_direction_i))\

                    print("added new point {}, midpoint between {} and {}, at index {}".format(midpoint, p_i, n_i, new_point_index))

                    adjacencies_by_point_index[new_point_index] = [None] * 6  # all new points will have 6 neighbors, only the 12 original vertices have 5
                    neighbor_direction_back_to_parent = get_opposite_neighbor_direction(neighbor_direction_i)

                    # pi_index_ni and ni_index_pi are the ACTUAL INDICES IN THE ADJACENCY LIST, which will DIFFER for the original 12 vertices because they only have 5 neighbors
                    # for interior (later-added) points, these will coincide with neighbor_direction_i and neighbor_direction_back_to_parent
                    # print("pi>ni = {} , ni>pi = {}".format(pi_index_ni, ni_index_pi))

                    # update the adjacencies by replacing the original neighbor with the midpoint
                    print("pre-replacement adjacency of parent {} = {}".format(p_i, adjacencies_by_point_index[p_i]))
                    pi_index_ni = adjacencies_by_point_index[p_i].index(n_i)
                    adjacencies_by_point_index[p_i][pi_index_ni] = new_point_index
                    adjacencies_by_point_index[new_point_index][neighbor_direction_i] = n_i  # here use direction, not index, in case parent has 5 neighbors
                    print("post-replacement adjacency of parent {} = {}".format(p_i, adjacencies_by_point_index[p_i]))

                    # the same must be done for the original neighbor as well
                    print("pre-replacement adjacency of neighbor {} = {}".format(n_i, adjacencies_by_point_index[n_i]))
                    ni_index_pi = adjacencies_by_point_index[n_i].index(p_i)
                    adjacencies_by_point_index[n_i][ni_index_pi] = new_point_index
                    adjacencies_by_point_index[new_point_index][neighbor_direction_back_to_parent] = p_i  # here use direction, not index, in case parent has 5 neighbors
                    print("post-replacement adjacency of neighbor {} = {}".format(n_i, adjacencies_by_point_index[n_i]))

                    print("post-replacement adjacency of new point {} = {}".format(new_point_index, adjacencies_by_point_index[new_point_index]))

                print("finished bisecting first three neighbors for point {}".format(p_i))

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
                print("filling out adjacencies for point {}: originally {}".format(p_i, adjacencies_p))
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
            
                print("- got filled neighbors for {}: {}".format(p_i, adjacencies_p))
                assert adjacencies_by_point_index[p_i] == adjacencies_p  # checking that modifying object by reference is working
                indices_filled_to_completion.append(p_i)
            
            for p_i in indices_filled_to_completion:
                # don't do this in the loop or it'll skip stuff
                indices_to_fill_out.remove(p_i)

            print("- finished filling out adjacencies for new points")
            assert len(indices_to_fill_out) == 0, "leftover points did not get filled out: {}".format(indices_to_fill_out)



            # dldl approach; this doesn't work, unfortunately
            # can use paths along known edges to work out the rest of the neighbors for a new point
            # know the parent point will be the neighbor in the opposite direction (e.g. create a DL neighbor, then parent is its UR neighbor)
            # e.g. a DL midpoint's L neighbor is found by the path (DL, U) (starting from the new point, go DL (to the original neighbor that it was created from), then U (the point that is a knight's jump 0.5*D + 1*L from the original parent point); easier to see graphically)
            # paths in this kind of lattice:
            # L = (known=L) from L; (DL, U) from DL; (U, DL) from D
            # DL = (L, D) from L; (known=DL) from DL; (D, L) from D
            # D = (R, DL) from L; (DL, R) from DL; (known=D) from D
            # R = (known=R) from L; (UR, D) from DL; (D, UR) from D
            # UR = (R, U) from L; (known=UR) from DL; (U, R) from D
            # U = (L, UR) from L; (UR, L) from DL; (known=U) from D

            # keep track of index_this_iteration_started_at (=10*2**(2*i)+2: 12, 42, 162, 642, etc.)
            # iterate from that index and fill in missing neighbors, all points starting at that index will have been new additions in this iteration
            # for p_i in l_new_point_indices:
            #     # should have L and R neighbors, indices 0 and 3
            #     adjacencies_list = adjacencies_by_point_index[p_i]
            #     print("filling adjacencies for point {}: {}".format(p_i, adjacencies_list))
            #     print("adjacencies of known neighbors:")
            #     for n_i in adjacencies_list:
            #         if n_i is not None:
            #             print("n_i {}: {}".format(n_i, adjacencies_by_point_index[n_i]))
            #     assert adjacencies_list[0] is not None and adjacencies_list[3] is not None, "expected pre-filled neighbor value from creation of L point index {}, but have {}".format(p_i, adjacencies_list)
            #     l = adjacencies_list[0]
            #     r = adjacencies_list[3]
            #     dl = adjacencies_by_point_index[l][2]  # (L, D)
            #     d = adjacencies_by_point_index[r][1]  # (R, DL)
            #     ur = adjacencies_by_point_index[r][5]  # (R, U)
            #     u = adjacencies_by_point_index[l][4]  # (L, UR)
            #     adjacencies_list = [l, dl, d, r, ur, u]
            #     print("got filled neighbors for {}: {}".format(p_i, adjacencies_list))
            #     assert adjacencies_by_point_index[p_i] == adjacencies_list  # checking that modifying object by reference is working
            #     indices_to_fill_out.remove(p_i)
            # for p_i in dl_new_point_indices:
            #     # should have DL and UR neighbors, indices 1 and 4
            #     adjacencies_list = adjacencies_by_point_index[p_i]
            #     print("filling adjacencies for point {}: {}".format(p_i, adjacencies_list))
            #     assert adjacencies_list[1] is not None and adjacencies_list[4] is not None, "expected pre-filled neighbor value from creation of DL point index {}, but have {}".format(p_i, adjacencies_list)
            #     raise
            #     indices_to_fill_out.remove(p_i)
            # for p_i in d_new_point_indices:
            #     # should have D and U neighbors, indices 2 and 5
            #     adjacencies_list = adjacencies_by_point_index[p_i]
            #     print("filling adjacencies for point {}: {}".format(p_i, adjacencies_list))
            #     assert adjacencies_list[2] is not None and adjacencies_list[5] is not None, "expected pre-filled neighbor value from creation of D point index {}, but have {}".format(p_i, adjacencies_list)
            #     raise
            #     indices_to_fill_out.remove(p_i)
            # assert len(indices_to_fill_out) == 0, "leftover points did not get filled out: {}".format(indices_to_fill_out)

            # old way, xyz4edge approach
            # new_adjacencies_xyz = {}
            # for existing_point, neighs in adjacencies_xyz.items():
            #     new_neighs = []
            #     for neigh in neighs:
            #         e = tuple(sorted((existing_point, neigh)))
            #         midpoint = midpoints[e]
            #         new_neighs.append(midpoint)
            #     new_adjacencies_xyz[existing_point] = new_neighs
            # for e in all_edges:
            #     new_point = midpoints[e]
            #     new_neighs = []
            #     v0, v1 = e
            #     new_neighs += [v0, v1]
            #     v0_neighs_old = adjacencies_xyz[v0]
            #     v1_neighs_old = adjacencies_xyz[v1]
            #     in_common = set(v0_neighs_old) & set(v1_neighs_old)
            #     assert len(in_common) == 2
            #     va, vb = in_common
            #     four_edges = [(v0, va), (v0, vb), (v1, va), (v1, vb)]
            #     for e4 in four_edges:
            #         e4 = tuple(sorted(e4))
            #         mp4 = midpoints[e4]
            #         new_neighs.append(mp4)
            #     new_adjacencies_xyz[new_point] = new_neighs
            # # done creating new adjacencies
            # adjacencies_xyz = new_adjacencies_xyz

            position_memo_fp = "/home/wesley/programming/Mapping/MemoIcosaPosition_Iteration{}.txt".format(iteration_i)
            adjacency_memo_fp = "/home/wesley/programming/Mapping/MemoIcosaAdjacency_Iteration{}.txt".format(iteration_i)
            print("writing memoization files")
            with open(position_memo_fp, "w") as f:
                for p_i in range(len(ordered_points)):
                    usp = ordered_points[p_i]
                    x, y, z = usp.get_coords("xyz")
                    lat, lon = usp.get_coords("latlondeg")
                    s = "{}:({},{},{}),({},{})\n".format(p_i, x, y, z, lat, lon)
                    f.write(s)
            with open(adjacency_memo_fp, "w") as f:
                for p_i in range(len(ordered_points)):
                    neighbor_indices_str = ",".join(str(x) for x in adjacencies_by_point_index[p_i])
                    s = "{}:{}\n".format(p_i, neighbor_indices_str)
                    f.write(s)
            print("- done writing memoization files")

            print("now have {} points, iteration {}".format(len(ordered_points), iteration_i))

        # convert to UnitSpherePoint
        # conversions = {}
        # for v in adjacencies_xyz:
        #     v_latlon = mcm.unit_vector_cartesian_to_lat_lon(*v)  # can parallelize this later by putting points in an array, but this part doesn't take that long so far, even for many Icosahedron points
        #     coords_dict = {"xyz": v, "latlondeg": v_latlon}
        #     usp = UnitSpherePoint(coords_dict)
        #     conversions[tuple(v)] = usp
        # adjacencies_usp = {}
        # for v0, neighs in adjacencies_xyz.items():
        #     neighs_usp = []
        #     for v1 in neighs:
        #         neighs_usp.append(conversions[v1])
        #     adjacencies_usp[conversions[v0]] = neighs_usp

        return ordered_points, adjacencies_by_point_index

        # TODO implement something like this
        # assert type(points_ordered) is list
        # assert all(type(x) is UnitSpherePoint for x in points_ordered)
        # assert type(adjacencies_point_indices) is dict
        # assert all(type(x) is int and type(y) is list for x, y in adjacencies_point_indices.items())
        # assert 0 in adjacencies_point_indices
        # assert all(type(x) is int for x in adjacencies_point_indices[0])
        # return points_ordered, adjacencies_point_indices


if __name__ == "__main__":
    edge_length_km = 250
    test_lattice = IcosahedralGeodesicLattice(edge_length_km)
    # test_lattice.plot_points()
    data = test_lattice.place_random_data()
    test_lattice.plot_data(data)
    
