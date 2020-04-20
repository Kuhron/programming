import numpy as np

from Lattice import Lattice


class IcosahedralGeodesicLattice(Lattice):
    EARTH_RADIUS_KM = 6371
    CADA_II_RADIUS_FACTOR = 2.27444
    CADA_II_RADIUS_KM = CADA_II_RADIUS_FACTOR * EARTH_RADIUS_KM

    def __init__(self, edge_length_km):
        self.edge_length_km = edge_length_km

    def get_adjacencies(self):
        # edge_length_km determines how high the resolution is

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
        neighbors_by_name = {
            "NP": ["NR0", "NRp72", "NRp144", "NRm72", "NRm144"],
            "NR0": ["NP", "NRp72", "NRm72", "SRp36", "SRm36"],
            "NRp72": ["NP", "NR0", "NRp144", "SRp36", "SRp108"],
            "NRp144": ["NP", "NRp72", "NRm144", "SRp108", "SR180"],
            "NRm72": ["NP", "NR0", "NRm144", "SRm36", "SRm108"],
            "NRm144": ["NP", "NRm72", "NRp144", "SRm108", "SR180"],
            "SR180": ["SP", "SRp108", "SRm108", "NRp144", "NRm144"],
            "SRp108": ["SP", "SR180", "SRp36", "NRp144", "NRp72"],
            "SRp36": ["SP", "SRp108", "SRm36", "NRp72", "NR0"],
            "SRm108": ["SP", "SR180", "SRm36", "NRm144", "NRm72"],
            "SRm36": ["SP", "SRm108", "SRp36", "NRm72", "NR0"],
            "SP": ["SR180", "SRp108", "SRp36", "SRm108", "SRm36"],
        }
        assert len(neighbors_by_name) == 12 and all(len(vals) == 5 for vals in neighbors_by_name.values())
        # check transitivity of neighborliness, since I input the lists manually
        for point_name, neighbors in neighbors_by_name.items():
            for neigh in neighbors:
                assert point_name in neighbors_by_name[neigh], "intransitive adjacency with {} and {}".format(point_name, neigh)
        # icosahedron points are correct now; time to divide edges

        adjacencies_xyz = {}
        for point_name, point_latlon in icosahedron_original_points_latlon.items():
            neighbor_latlons_lst = []
            for neighbor_name in neighbors_by_name[point_name]:
                neighbor_latlons_lst.append(icosahedron_original_points_latlon[neighbor_name])
            point_xyz = mcm.unit_vector_lat_lon_to_cartesian(*point_latlon)
            neighbor_xyz_lst = [tuple(mcm.unit_vector_lat_lon_to_cartesian(*n)) for n in neighbor_latlons_lst]
            adjacencies_xyz[tuple(point_xyz)] = neighbor_xyz_lst

        # bisect edges until reach resolution
        iteration_i = 0
        while True:
            random_point = random.choice(list(adjacencies_xyz.keys()))
            neigh = random.choice(adjacencies_xyz[random_point])
            # v0 = mcm.unit_vector_lat_lon_to_cartesian(*random_point)
            # v1 = mcm.unit_vector_lat_lon_to_cartesian(*neigh)
            angle_radians = mcm.angle_between_vectors(random_point, neigh)
            edge_length = cada_ii_radius_km * angle_radians
            print("edge_length = {} km, iteration {}".format(edge_length, iteration_i))
            if edge_length <= edge_length_km:
                break

            # bisection and neighbor updating
            # start by getting all edges
            all_edges = set()
            for v0, neighs in adjacencies_xyz.items():
                for v1 in neighs:
                    e = tuple(sorted((v0, v1)))
                    all_edges.add(e)
            
            # then get midpoint of each edge, still indexed by endpoints
            midpoints = {e: None for e in all_edges}
            for e in all_edges:
                # midpoint is normalized average
                v0, v1 = e
                midpoint_raw = (np.array(v0)+np.array(v1))/2
                midpoint = tuple((midpoint_raw / np.linalg.norm(midpoint_raw)).reshape(3))
                midpoints[e] = midpoint

            # then redo neighbor array. existing point neighbors -> midpoints of all its edges (whether there are 5 or 6)
            # new point neighbors -> endpoints of its edge (2) + midpoints of the edges from its endpoints to the two neighbors they have in common (in the OLD adjacencies) (4), so all new points have 6 new neighbors
            new_adjacencies_xyz = {}
            for existing_point, neighs in adjacencies_xyz.items():
                new_neighs = []
                for neigh in neighs:
                    e = tuple(sorted((existing_point, neigh)))
                    midpoint = midpoints[e]
                    new_neighs.append(midpoint)
                new_adjacencies_xyz[existing_point] = new_neighs
            for e in all_edges:
                new_point = midpoints[e]
                new_neighs = []
                v0, v1 = e
                new_neighs += [v0, v1]
                v0_neighs_old = adjacencies_xyz[v0]
                v1_neighs_old = adjacencies_xyz[v1]
                in_common = set(v0_neighs_old) & set(v1_neighs_old)
                assert len(in_common) == 2
                va, vb = in_common
                four_edges = [(v0, va), (v0, vb), (v1, va), (v1, vb)]
                for e4 in four_edges:
                    e4 = tuple(sorted(e4))
                    mp4 = midpoints[e4]
                    new_neighs.append(mp4)
                new_adjacencies_xyz[new_point] = new_neighs
            # done creating new adjacencies
            adjacencies_xyz = new_adjacencies_xyz

            print("now have {} points, iteration {}".format(len(adjacencies_xyz), iteration_i))
            iteration_i += 1

        # test: plot them
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ps = list(adjacencies_xyz.keys())
        xs = [p[0] for p in ps]
        ys = [p[1] for p in ps]
        zs = [p[2] for p in ps]
        ax.scatter(xs, ys, zs)
        plt.show()

        # convert to latlon
        conversions = {}
        for v in adjacencies_xyz:
            latlon = mcm.unit_vector_cartesian_to_lat_lon(*v)
            conversions[tuple(v)] = tuple(latlon)
        adjacencies_latlon = {}
        for v0, neighs in adjacencies_xyz.items():
            neighs_latlon = []
            for v1 in neighs:
                neighs_latlon.append(conversions[v1])
            adjacencies_latlon[conversions[v0]] = neighs_latlon
        return adjacencies_latlon
