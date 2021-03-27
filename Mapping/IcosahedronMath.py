# try to decrease the amount of memory used up by large lattices
# keep info about Icosa coordinates in files, look things up in tables, rather than keeping it all in RAM


import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from UnitSpherePoint import UnitSpherePoint


def get_latlon_from_point_number(point_number):
    raise NotImplementedError


def get_xyz_from_point_number(point_number):
    raise NotImplementedError


def get_iterations_from_points(n):
    try:
        # n_points(n_iters) = 10 * (4**n_iters) + 2
        return {12: 0, 42: 1, 162: 2, 642: 3, 2562: 4, 10242: 5}[n]
    except KeyError:
        iterations = get_exact_iterations_from_points(n)
        assert iterations % 1 == 0, "number of points {} gave non-int number of iterations; make sure it is 2+10*(4**n)".format(n)
        return iterations


def get_points_from_iterations(n):
    try:
        return {0: 12, 1: 42, 2: 162, 3: 642, 4: 2562, 5: 10242}[n]
    except KeyError:
        points = get_exact_points_from_iterations(n)
        assert points % 1 == 0, "number of iterations {} gave non-int number of points {}".format(n, points)
        return points


def get_exact_points_from_iterations(n_iters):
    return 2 + 10 * (4 ** n_iters)


def get_exact_iterations_from_points(n_points):
    return np.log((n_points - 2)/10) / np.log(4)  # just use change of base since np.log(arr, b) doesn't like arrays


def get_iterations_needed_for_point_number(point_number):
    # n_points is at least point_number+1, e.g. if it's point number 12, you actually have 13 points
    n_points_min = point_number + 1
    if n_points_min <= 12:
        return 0  # the baseline icosa has 12 vertices
    iters_exact = get_exact_iterations_from_points(n_points_min)
    return math.ceil(iters_exact)


def get_specific_adjacencies(point_numbers, n_iterations):
    verify_valid_point_numbers(point_numbers, n_iterations)
    memo_fp = get_adjacency_memo_fp(n_iterations)
    with open(memo_fp) as f:
        lines = f.readlines()
    lines = [lines[i] for i in point_numbers]
    d = {}
    for point_number, l in zip(point_numbers,lines):
        pi, neighbors = parse_adjacency_line(l)
        assert pi == point_number
        d[pi] = neighbors
    return d


def get_specific_positions(point_numbers, n_iterations):
    verify_valid_point_numbers(point_numbers, n_iterations)
    memo_fp = get_position_memo_fp(n_iterations)
    with open(memo_fp) as f:
        lines = f.readlines()
    lines = [lines[i] for i in point_numbers]
    d = {}
    for point_number, l in zip(point_numbers,lines):
        pi, xyz, latlon = parse_position_line(l)
        assert pi == point_number
        d[pi] = {"xyz":xyz, "latlondeg":latlon}
    return d


def verify_valid_point_numbers(point_numbers, n_iterations):
    points_at_iter = get_exact_points_from_iterations(n_iterations)
    for p in point_numbers:
        if type(p) is not int:
            raise TypeError("invalid point number, expected int: {}".format(p))
        if p < 0:
            raise ValueError("point number should be non-negative: {}".format(p))
        if p >= points_at_iter:
            raise ValueError("point number {} too high for {} iterations, which has {} points".format(p, n_iterations, points_at_iter))


def parse_adjacency_line(l):
    pi, neighbors_str = l.strip().split(":")
    pi = int(pi)
    neighbors = [int(x) for x in neighbors_str.split(",")]
    return pi, neighbors


def parse_position_line(l):
    pi, rest = l.strip().split(":")
    pi = int(pi)
    xyz_str, latlon_str = rest.split(";")
    xyz = [float(x) for x in xyz_str.split(",")]
    assert len(xyz) == 3
    latlon = [float(x) for x in latlon_str.split(",")]
    assert len(latlon) == 2
    return pi, xyz, latlon


def parse_adjacency_memo_file(memo_fp):
    # format: each line is index:neighbor_list (comma separated point numbers)
    # e.g. 0:1752,1914,2076,2238,2400
    with open(memo_fp) as f:
        lines = f.readlines()
    d = {}
    for l in lines:
        pi, neighbors = parse_adjacency_line(l)
        d[pi] = neighbors
    return d


def parse_position_memo_file(memo_fp):
    # format: each line is index:xyz;latlon (xyz and latlon are comma-separated)
    # e.g. 0:6.123233995736766e-17,0.0,1.0;90,0
    with open(memo_fp) as f:
        lines = f.readlines()
    d = {}
    for l in lines:
        pi, xyz, latlon = parse_position_line(l)
        d[pi] = {"xyz":xyz, "latlondeg":latlon}
    return d


def plot_neighbor_relationships(n_iterations):
    # hopefully at some point I can figure out a mathematical expression for all of this and not have to memoize anything
    d = get_adjacency_memo_dict(n_iterations)
    n_points = get_exact_points_from_iterations(n_iterations)
    point_numbers = range(12, n_points)
    neighbor_indices = range(6)
    colors = ["red","yellow","green","blue","purple","black"]
    for ni, c in zip(neighbor_indices, colors):
        neighbor_numbers_at_index = [d[pi][ni] for pi in point_numbers]
        plt.scatter(point_numbers, neighbor_numbers_at_index, color=c, alpha=0.4)
    plt.show()


def plot_xyzs(n_iterations):
    d = get_position_memo_dict(n_iterations)
    n_points = get_exact_points_from_iterations(n_iterations)
    point_numbers = range(12, n_points)
    xs = [d[pi]["xyz"][0] for pi in point_numbers]
    ys = [d[pi]["xyz"][1] for pi in point_numbers]
    zs = [d[pi]["xyz"][2] for pi in point_numbers]

    plt.scatter(point_numbers, xs)
    plt.title("x")
    plt.show()

    plt.scatter(point_numbers, ys)
    plt.title("y")
    plt.show()

    plt.scatter(point_numbers, zs)
    plt.title("z")
    plt.show()


def plot_latlons(n_iterations):
    d = get_position_memo_dict(n_iterations)
    n_points = get_exact_points_from_iterations(n_iterations)
    point_numbers = range(12, n_points)
    lats = [d[pi]["latlondeg"][0] for pi in point_numbers]
    lons = [d[pi]["latlondeg"][1] for pi in point_numbers]
    
    plt.scatter(point_numbers, lats)
    plt.title("lat")
    plt.show()

    plt.scatter(point_numbers, lons)
    plt.title("lon")
    plt.show()


def plot_coordinate_patterns(n_iterations):
    # for trying to get some pattern recognition and figure out what the functions are that determine the positions and adjacencies of the icosahedron points
    # right now it seems pretty hopeless; there are a lot of complicated patterns, they look cool but I don't understand them
    # the X plot has Sierpinski fractals, lots of other fractal structures visible at high iteration numbers (~7)
    plot_neighbor_relationships(n_iterations)
    plot_xyzs(n_iterations)
    plot_latlons(n_iterations)


def get_adjacency_memo_fp(n_iterations):
    return "/home/wesley/programming/Mapping/MemoIcosa/MemoIcosaAdjacency_Iteration{}.txt".format(n_iterations)


def get_position_memo_fp(n_iterations):
    return "/home/wesley/programming/Mapping/MemoIcosa/MemoIcosaPosition_Iteration{}.txt".format(n_iterations)


def get_adjacency_memo_dict(n_iterations):
    return parse_adjacency_memo_file(get_adjacency_memo_fp(n_iterations))


def get_position_memo_dict(n_iterations):
    return parse_position_memo_file(get_position_memo_fp(n_iterations))


def get_starting_points_latlon_named():
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
    return icosahedron_original_points_latlon


def get_starting_points_adjacency_named():
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

    return original_points_neighbors_by_name


def get_starting_points():
    icosahedron_original_points_latlon = get_starting_points_latlon_named()
    original_points_neighbors_by_name = get_starting_points_adjacency_named()

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

    return ordered_points, adjacencies_by_point_index


def get_sample_average_edge_length(points, adjacencies, radius):
    # check some random edges to get average edge length
    edge_lengths = []
    for _ in range(100):
        random_point_index = random.choice(list(adjacencies.keys()))
        neighbor_index = random.choice(adjacencies[random_point_index])
        p0 = points[random_point_index]
        p1 = points[neighbor_index]
        angle_radians = UnitSpherePoint.get_angle_radians_between(p0, p1)
        edge_length = radius * angle_radians
        edge_lengths.append(edge_length)
    edge_length = np.mean(edge_lengths)
    return edge_length


def get_iterations_needed_for_edge_length(edge_length, radius):
    # edge_length_km determines how high the resolution is
    initial_edge_length = get_icosa_edge_length_from_radius_to_vertex(radius)  # edge length at iteration 0, when it's just the 12 vertices
    factor = initial_edge_length / edge_length
    # each iteration halves the edge length
    iterations_needed = int(np.ceil(np.log2(factor)))
    return iterations_needed


def get_icosa_edge_length_from_radius_to_vertex(r):
    # derive from the inverse formula at https://en.wikipedia.org/wiki/Regular_icosahedron
    # radius of sphere that touches icosa at all vertices = (edge_length)/4 * sqrt(10 + 2*sqrt(5))
    return r * 4 / np.sqrt(10 + 2 * np.sqrt(5))


def get_opposite_neighbor_direction(i):
    # call the directions (on rectangle representation) L, DL, D, R, UR, U (in counterclockwise order, as they appear on the rectangle representation for a generic peel-internal point)
    return {0: 3, 3: 0, 1: 4, 4: 1, 2: 5, 5: 2}[i]  # map L vs R, DL vs UR, D vs U
    # more succinctly could do return (i+3)%6, but the dict makes it more readable and also throws for unexpected stuff like 1.5 or -1


def get_parent_point(point_number):
    # each point except the initial 12 is created from a "parent", a pre-existing point from one of the previous iterations
    # at each iteration, each existing point except the poles gets three new children
    # so e.g. iteration 1 has 42 points, 40 of those get 3 children each, creating 120 new points, so iteration 2 has 162 points, correct
    # so each new generation skips 0 and 1 (the poles) and starts with point #2 in creating the children
    # e.g. from gen 1 to 2, start with point 2, create points 42,43,44, then 3>45,46,47, ..., 41>159,160,161
    if point_number < 12:
        return None  # initial points have no parents
    raise Exception("doesn't work, need to prove it")
    return point_number//3 - 12  # not sure this will always work, TODO prove it


def get_parent_point_direction_label(point_number):
    # 2's first children are 42,43,44, which look back to 2 in the directions of R, UR, U respectively
    # these are the only three directions that a parent can be found in, and they will always happen in this order
    m = point_number % 3
    # for 42, m is 0, direction is R; 43 1 UR; 44 2 U
    return ["R", "UR", "U"][m]


def get_parent_point_direction_number(point_number):
    return get_direction_number_from_label(get_parent_point_direction_label(point_number))


def get_direction_number_from_label(s):
    return ["L", "DL", "D", "R", "UR", "U"].index(s)


def get_direction_label_from_number(i):
    return ["L", "DL", "D", "R", "UR", "U"][i]



if __name__ == "__main__":
    point_number = random.randint(12, 655362-1)
    n_iterations = get_iterations_needed_for_point_number(point_number)
    # n_points = get_exact_points_from_iterations(n_iterations)
    # plot_coordinate_patterns(n_iterations)

    # point_numbers = np.random.randint(0, n_points, (100,))
    point_numbers = [point_number]
    adj = get_specific_adjacencies(point_numbers, n_iterations)[point_number]
    print(adj)
    # print(get_specific_positions(point_numbers, n_iterations))
    print(get_parent_point(point_number))
    parent_point_direction_number = get_parent_point_direction_number(point_number)
    print(adj[parent_point_direction_number])
