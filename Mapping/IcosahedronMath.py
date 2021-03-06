# try to decrease the amount of memory used up by large lattices
# keep info about Icosa coordinates in files, look things up in tables, rather than keeping it all in RAM


import math
import functools
import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import MapCoordinateMath as mcm
from UnitSpherePoint import UnitSpherePoint



EARTH_RADIUS_KM = 6371
CADA_II_RADIUS_FACTOR = 2.116
CADA_II_RADIUS_KM = CADA_II_RADIUS_FACTOR * EARTH_RADIUS_KM


def get_latlon_from_point_number(point_number):
    pos = get_position_recursive(point_number)
    return pos["latlondeg"]


def get_latlons_from_point_numbers(point_numbers):
    poses = get_positions_recursive(point_numbers)
    return [pos["latlondeg"] for pos in poses]


def get_xyz_from_point_number(point_number):
    pos = get_position_recursive(point_number)
    return pos["xyz"]


def get_xyzs_from_point_numbers(point_numbers):
    poses = get_positions_recursive(point_numbers)
    return [pos["xyz"] for pos in poses]


def get_usp_from_point_number(point_number):
    pos = get_position_recursive(point_number)
    assert type(pos) is dict
    return UnitSpherePoint(pos, point_number=point_number)


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


@functools.lru_cache(maxsize=10000)
def get_exact_iterations_from_points(n_points):
    return np.log((n_points - 2)/10) / np.log(4)  # just use change of base since np.log(arr, b) doesn't like arrays


def get_iterations_needed_for_point_number(point_number):
    # n_points is at least point_number+1, e.g. if it's point number 12, you actually have 13 points
    n_points_min = point_number + 1
    if n_points_min <= 12:
        return 0  # the baseline icosa has 12 vertices
    iters_exact = get_exact_iterations_from_points(n_points_min)
    return math.ceil(iters_exact)


def get_specific_adjacencies_from_memo(point_numbers, n_iterations):
    verify_valid_point_numbers(point_numbers, n_iterations)
    memo_fp = get_adjacency_memo_fp(n_iterations)
    # print("getting adjacency memo for iteration {}: {}".format(n_iterations, memo_fp))
    with open(memo_fp) as f:
        notify_memo_accessed(memo_fp)
        lines = f.readlines()
    lines = [lines[i] for i in point_numbers]
    d = {}
    for point_number, l in zip(point_numbers,lines):
        pi, neighbors = parse_adjacency_line(l)
        assert pi == point_number
        d[pi] = neighbors
    # print("returning {}".format(d))
    return d


def get_specific_adjacency_from_memo(point_number, n_iterations):
    return get_specific_adjacencies_from_memo([point_number], n_iterations)[point_number]


def get_specific_positions_from_memo(point_numbers, n_iterations):
    verify_valid_point_numbers(point_numbers, n_iterations)
    memo_fp = get_position_memo_fp(n_iterations)
    with open(memo_fp) as f:
        notify_memo_accessed(memo_fp)
        lines = f.readlines()
    lines = [lines[i] for i in point_numbers]
    d = {}
    for point_number, l in zip(point_numbers,lines):
        pi, xyz, latlon = parse_position_line(l)
        assert pi == point_number
        d[pi] = {"xyz":xyz, "latlondeg":latlon}
    return d


def get_specific_position_from_memo(point_number):
    n_iterations = get_iterations_needed_for_point_number(point_number)
    return get_specific_positions_from_memo([point_number], n_iterations)[point_number]


def verify_valid_point_numbers(point_numbers, n_iterations):
    points_at_iter = get_exact_points_from_iterations(n_iterations)
    for p in point_numbers:
        if type(p) is not int:
            raise TypeError("invalid point number, expected int: {}".format(p))
        if p < 0:
            raise ValueError("point number should be non-negative: {}".format(p))
        if p >= points_at_iter:
            raise ValueError("point number {} too high for {} iterations, which has {} points".format(p, n_iterations, points_at_iter))


def verify_can_have_children(point_number, n_iterations):
    if point_number in [0, 1]:
        raise ValueError("point {} cannot have children".format(point_number))
    # point cannot have children in the same iteration when it is born, so require that the point existed one iteration ago (or earlier)
    verify_valid_point_numbers([point_number], n_iterations-1)


def is_valid_point_number(point_number, n_iterations):
    try:
        verify_valid_point_numbers([point_number], n_iterations)
    except ValueError:
        return False
    return True


def can_have_children(point_number, n_iterations):
    try:
        verify_can_have_children(point_number, n_iterations)
    except ValueError:
        return False
    return True


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


def get_adjacency_line(point_number, ordered_neighbor_point_numbers):
    s1 = str(point_number)
    s2 = ",".join(str(x) for x in ordered_neighbor_point_numbers)
    return s1 + ":" + s2 + "\n"


def get_position_line(point_number, xyz, latlon):
    s1 = str(point_number)
    s2 = ",".join(str(x) for x in xyz)
    s3 = ",".join(str(x) for x in latlon)
    return s1 + ";" + s2 + ";" + s3 + "\n"


def parse_adjacency_memo_file(memo_fp):
    # format: each line is index:neighbor_list (comma separated point numbers)
    # e.g. 0:1752,1914,2076,2238,2400
    with open(memo_fp) as f:
        notify_memo_accessed(memo_fp)
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
        notify_memo_accessed(memo_fp)
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


def get_original_points_order_by_name():
    original_points_order_by_name = [
        "NP", "SP",  # poles
        "NR0", "SRp36",  # peel 0
        "NRp72", "SRp108",  # peel 1
        "NRp144", "SR180",  # peel 2
        "NRm144", "SRm108",  # peel 3
        "NRm72", "SRm36",  # peel 4
    ]
    return original_points_order_by_name


def get_starting_points():
    # print("getting starting icosa points")
    icosahedron_original_points_latlon = get_starting_points_latlon_named()
    original_points_neighbors_by_name = get_starting_points_adjacency_named()

    # put them in this ordering convention:
    # north pole first, south pole second, omit these from all expansion operations, by only operating on points[2:] (non-pole points)
    # new points are appended to the point list in the order they are created
    # order the neighbors in the following order, and only bisect the edges from each point to the first three:
    # - [north, west, southwest, (others)]. order of others doesn't matter that much, can just keep going counterclockwise
    # ordering of neighbors for poles thus doesn't matter as that list will never be used for expansion

    original_points_order_by_name = get_original_points_order_by_name()

    # keep the point objects in a single array that can be indexed by point index
    # the rest of the data, i.e., the adjacencies dictionary, should be all in terms of integer indices that refer to the points array

    ordered_points = []
    adjacencies_by_point_index = [None for i in range(12)]

    # place original points in the list
    for point_number, p_name in enumerate(original_points_order_by_name):
        point_index = len(ordered_points)
        p_latlon = icosahedron_original_points_latlon[p_name]
        p_xyz = mcm.unit_vector_lat_lon_to_cartesian(*p_latlon)
        coords_dict = {"xyz": p_xyz, "latlondeg": p_latlon}
        usp = UnitSpherePoint(coords_dict, point_number)
        ordered_points.append(usp)
    assert len(ordered_points) == 12, "initial icosa needs 12 vertices"

    # add their neighbors by index
    for point_index in range(len(ordered_points)):
        point_name = original_points_order_by_name[point_index]
        neighbor_names = original_points_neighbors_by_name[point_name]
        neighbor_indices = [original_points_order_by_name.index(name) for name in neighbor_names]
        adjacencies_by_point_index[point_index] = neighbor_indices
        # print("adjacencies now:\n{}\n".format(adjacencies_by_point_index))

    # print("done getting initial icosa points")
    return ordered_points, adjacencies_by_point_index


def get_starting_points_immutable():
    ordered_points, adj = get_starting_points()
    assert type(ordered_points) is list
    assert all(type(x) is UnitSpherePoint for x in ordered_points)
    ordered_points = tuple(ordered_points)
    assert type(adj) is list
    new_adj = ()
    for x in adj:
        assert type(x) is list
        assert all(type(y) is int for y in x)
        x_tup = tuple(x)
        extend_tup = (x_tup,)
        new_adj = new_adj + extend_tup
    return (ordered_points, new_adj)


def get_starting_point_neighbor_identity(point_number):
    # for 0 and 1 (the poles) this is still weird, it's not clear what the directions (L,DL,D,R,UR,U) would mean for them, ill-defined like east of the south pole
    # but for the other 10 starting points, there are five neighbors but one of them acts like two directions
    # e.g. on the northern ring, from the perspective of the peel below (west of) the point, the L neighbor is the north pole
    # but from the perspective of the peel above (east of) the point, the U neighbor is the north pole
    d = {}
    assert type(point_number) is int, point_number
    assert 2 <= point_number < 12, "invalid point for neighbor identity: {}".format(point_number)
    ring = get_starting_point_ring(point_number)
    if ring == "northern_ring":
        return ("L", "U")
    elif ring == "southern_ring":
        return ("D", "R")
    else:
        raise ValueError("invalid ring {}".format(ring))


def get_starting_point_ring(starting_point):
    original_points_order_by_name = get_original_points_order_by_name()
    ring_code = original_points_order_by_name[starting_point][:2]
    if ring_code == "NP":
        return "north_pole"
    elif ring_code == "SP":
        return "south_pole"
    elif ring_code == "NR":
        return "northern_ring"
    elif ring_code == "SR":
        return "southern_ring"
    else:
        raise ValueError("invalid ring code {}".format(ring_code))


def write_initial_memo_files():
    # ordered_points, adjacencies_by_point_index = get_starting_points()
    ordered_points = STARTING_POINTS_ORDERED
    adjacencies_by_point_index = STARTING_POINTS_ADJACENCY

    s_adj = ""
    s_pos = ""
    for pi, p in enumerate(ordered_points):
        adj = adjacencies_by_point_index[pi]
        l_adj = get_adjacency_line(pi, adj)
        l_pos = get_position_line(pi, p.xyz(), p.latlondeg())
        assert l_adj[-1] == l_pos[-1] == "\n"
        s_adj += l_adj
        s_pos += l_pos
    with open(get_adjacency_memo_fp(n_iterations=0), "w") as f:
        f.write(s_adj)
    with open(get_position_memo_fp(n_iterations=0), "w") as f:
        f.write(s_pos)
    print("initial memo files written")


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


@functools.lru_cache(maxsize=10000)
def get_child(parent, child_index, iteration):
    if parent in [0, 1]:
        raise ValueError("point {} cannot have children".format(parent))
    verify_can_have_children(parent, iteration)  # make sure parent exists and is old enough to have children
    adder = get_3adder_for_iteration(iteration)
    return 3 * (parent + adder) + child_index


@functools.lru_cache(maxsize=10000)
def get_parent(point_number):
    # each point except the initial 12 is created from a "parent", a pre-existing point from one of the previous iterations
    # at each iteration, each existing point except the poles gets three new children
    # so e.g. iteration 1 has 42 points, 40 of those get 3 children each, creating 120 new points, so iteration 2 has 162 points, correct
    # so each new generation skips 0 and 1 (the poles) and starts with point #2 in creating the children
    # e.g. from gen 1 to 2, start with point 2, create points 42,43,44, then 3>45,46,47, ..., 41>159,160,161
    if point_number < 12:
        return None  # initial points have no parents
    # see IcosaParentChildRelations.ods for math
    return point_number // 3 - get_3adder_for_iteration(get_iteration_born(point_number))


@functools.lru_cache(maxsize=10000)
def get_directional_parent(point_number):
    # safe but slow: use known process of creation of new points, guarantees correct answer
    return get_directional_parent_via_inheritance(point_number)
    # risky but fast: use patterns seen in the numbers, but without proof that these generalizations will always hold
    # return get_directional_parent_via_numerology(point_number)


@functools.lru_cache(maxsize=10000)
def get_directional_parent_via_inheritance(point_number):
    # the parent at the other end of the edge that was bisected to produce this point
    if point_number < 12:
        return None
    parent = get_parent(point_number)
    iteration_born = get_iteration_born(point_number)
    # print("{} was born i={}".format(point_number, iteration_born))
    parent_adjacency_in_born_iteration = get_adjacency_recursive(parent, iteration_born)
    parent_adjacency_in_previous_iteration = get_adjacency_recursive(parent, iteration_born-1)
    # print("parent previous adjacency: {}".format(parent_adjacency_in_previous_iteration))
    # print("parent current adjacency: {}".format(parent_adjacency_in_born_iteration))
    index_of_this_point_from_parent = parent_adjacency_in_born_iteration.index(point_number)
    previous_point_at_that_index = parent_adjacency_in_previous_iteration[index_of_this_point_from_parent]
    return previous_point_at_that_index
    # this avoids problems with indexing directions from the 12 initial points which only have 5 adjacencies with ill-defined directions


@functools.lru_cache(maxsize=10000)
def get_directional_parent_via_numerology(point_number):
    return NotImplemented
    # ?
    # easy class: they go up in order for each of the previous generation's points
    # ?
    # hard class: after that, they do some more chaotic things
    # ?



def test_directional_parent_correctness(point_numbers):
    failures = []
    for p in point_numbers:
        dp_inheritance = get_directional_parent_via_inheritance(p)
        dp_numerology = get_directional_parent_via_numerology(p)
        if dp_inheritance != dp_numerology:
            failure_tup = (p, dp_inheritance, dp_numerology)
    if len(failures) == 0:
        print("success! directional parents all agreed")
    else:
        for p, dp_i, dp_n in failures:
            print(f"failure: point {p} has directional parent of {dp_i} by inheritance (which is the correct one), but numerology gave {dp_n}")


def get_parents(point_number):
    p0 = get_parent(point_number)
    p1 = get_directional_parent(point_number)
    return [p0, p1]


def get_parent_chain(point_number):
    # go forward in time, starting from the initial point which gives rise ultimately to this one
    # return list of tuples, one for each iteration starting with zero up to and including the one where this point was born
    # tuples are (iteration, parent, child_index, child) where parent is reproducing this generation and child is born this generation
    if point_number < 12:
        iteration_born = 0
        parent = None
        child_index = None
        child = point_number
        tup = (iteration_born, parent, child_index, child)
        return [tup]
    else:
        iteration_born = get_iteration_born(point_number)
        child_index = get_child_index(point_number)
        parent = get_parent(point_number)
        parent_iteration_born = get_iteration_born(parent)
        previous_chain = get_parent_chain(parent)
        chain = [x for x in previous_chain]
        iterations_with_same_parent = list(range(parent_iteration_born+1, iteration_born))
        for i in iterations_with_same_parent:
            this_child_index = None
            this_child = None
            tup = (i, parent, this_child_index, this_child)
            chain.append(tup)
        
        # tup for this point
        final_tup = (iteration_born, parent, child_index, point_number)
        chain.append(final_tup)
        assert len(chain) == iteration_born + 1
        return chain


def get_ancestor_tree(point_number, existing_ancestry=None):
    # dict of child: parents
    if existing_ancestry is None:
        existing_ancestry = {}
    else:
        # for combining ancestry trees of multiple points, if we see a point's parent>child tuple already there,
        # then don't have to recalculate the rest of its ancestry which should also already be there
        pass

    ancestry = {}
    # use sets of tuples to take advantage of constant-time lookup
    farthest_back_generation = [point_number]
    while len(farthest_back_generation) > 0:
        new_farthest_back_generation = []
        for child in farthest_back_generation:
            if child in existing_ancestry or child in ancestry:
                # already know its parent
                continue
            parents = get_parents(child)
            parents = [x if x is not None else -1 for x in parents]  # convert to -1 for int sorting
            ancestry[child] = parents
            new_farthest_back_generation += parents
        farthest_back_generation = [x for x in new_farthest_back_generation if x != -1]
    return ancestry


def get_ancestor_tree_for_multiple_points(point_numbers):
    print(f"getting ancestor tree for {len(point_numbers)} points")
    ancestry = {}
    for i, p in enumerate(point_numbers):
        if i % 100 == 0:
            print(f"progress: {i}/{len(point_numbers)}")
        if p in ancestry:
            # don't need to call it
            continue
        else:
            p_ancestry = get_ancestor_tree(p, existing_ancestry=ancestry)
            ancestry.update(p_ancestry)
            assert p in ancestry, "failed to update ancestry correctly, should be adding parents of current point"
    print(f"done getting ancestor tree for {len(point_numbers)} points")
    return ancestry


def get_ancestor_graph(point_number):
    ancestry = get_ancestor_tree(point_number)
    return get_ancestor_graph_from_ancestor_tree(ancestry)


def get_ancestor_graph_for_multiple_points(point_numbers):
    ancestry = get_ancestor_tree_for_multiple_points(point_numbers)
    return get_ancestor_graph_from_ancestor_tree(ancestry)


def get_ancestor_graph_from_ancestor_tree(ancestry):
    g = nx.DiGraph()
    for child, (p0, p1) in ancestry.items():
        if p0 != -1:
            g.add_edge(p0, child)
        if p1 != -1:
            g.add_edge(p1, child)
    return g


def get_all_positions_in_ancestor_tree(ancestry):
    # auxiliary function meant to help make it easier to get the positions of a large number of points
    # by taking advantage of the fact that many of them share ancestry, so that position info can be reused without recalculation
    child_to_parents = ancestry
    pn_to_position = {}
    for child in sorted(child_to_parents.keys()):
        # do the lower-number points first because they are created earlier, and then later points can use position information from them
        p0, p1 = child_to_parents[child]
        # print(f"getting position from ancestry for child {child} of parents {p0}, {p1}")
        if p0 == -1 and p1 == -1:
            # print(f"getting position for parentless point {child}")
            pos = get_position_recursive(child)
            pn_to_position[child] = pos
        else:
            # print(f"getting position for point {child} with parents in the tree")
            pos0 = pn_to_position[p0]
            pos1 = pn_to_position[p1]
            # print(f"p0 is at {pos0}\np1 is at {pos1}")
            pos = get_position_of_child_from_parent_positions(pos0, pos1)
            # print(f"child is at {pos}")
            pn_to_position[child] = pos
    return pn_to_position


def is_parent_and_child(parent, child):
    return parent == get_parent(child)


def is_parent_and_child_direction(a, b, a_adjacency):
    # returns whether b is in a child-like direction from a's perspective
    # assert can_have_children(a, iteration)  # this is NOT necessary; it's just about DIRECTION, not actual children
    # however, if a is a pole, need to either raise or return False (it can't be a parent)
    assert a != b, "cannot check child-directionality from point {} to itself; check that this was intended".format(a)
    # print("checking if {}>{} is parent to child direction at i={}".format(a, b, iteration))
    if a in [0, 1]:
        # print("a is a pole, returning that child-directionality is False")
        return False
        # raise ValueError("can't get child-like direction from the poles; point number is {}".format(a))
    b_index = a_adjacency.index(b)
    res = b_index in [0,1,2]
    # print("found {} at index {} in a_adj {}, child-directionality is {}".format(b, b_index, a_adj, res))
    return res


def unify_five_and_six(adjacency, point_number):
    # converts the five-point adjacencies for points 2-11 (inclusive) into six-point, where two of the points are the same according to the neighbor identities for those 10 points
    if point_number >= 12:
        assert len(adjacency) == 6
        return adjacency
    elif point_number < 2:
        raise Exception("cannot unify adjacency to six-point for point number {}".format(point_number))
    elif len(adjacency) == 6:
        # it's already in 6-point form, return it
        return adjacency

    identity_pair = get_starting_point_neighbor_identity(point_number)
    idx, idy = identity_pair
    idx_int = get_direction_number_from_label(idx)
    idy_int = get_direction_number_from_label(idy)
    # what order are the neighbors in originally? want a dict or something for better ease of constructing the faux-6-neighbor list
    # they always start with L and go counterclockwise
    # so a point on northern ring will be L/U,DL,D,R,UR, need to add L/U at end
    # and a point on southern ring will be L,DL,D/R,UR,U, need to add another D/R after the first one
    if identity_pair == ("L", "U"):
        # northern ring
        assert idx_int == 0 and idy_int == 5
        adj = adjacency + [adjacency[0]]
    elif identity_pair == ("D", "R"):
        # southern ring
        assert idx_int == 2 and idy_int == 3
        adj = adjacency[:3] + [adjacency[2]] + adjacency[3:]
    else:
        raise ValueError("invalid identity pair {}".format(identity_pair))

    return adj


@functools.lru_cache(maxsize=10000)
def get_adjacency_recursive(point_number, iteration):
    # use get_adjacency_when_born() here as base case
    # for non-born iterations, use the formula for child number from parent, index, and iteration
    # print("getting adjacency recursive for p#{} in i#{}".format(point_number, iteration))

    if iteration == get_iteration_born(point_number):
        return get_adjacency_when_born(point_number)

    if point_number in [0, 1]:
        if iteration == 0:
            ordered_points, adj = STARTING_POINTS
            return adj[point_number]
        elif point_number == 0:
            previous_adj = get_adjacency_recursive(point_number, iteration-1)
            return [get_north_pole_neighbor(previous_neighbor, iteration) for previous_neighbor in previous_adj]
        elif point_number == 1:
            previous_adj = get_adjacency_recursive(point_number, iteration-1)
            return [get_south_pole_neighbor(previous_neighbor, iteration) for previous_neighbor in previous_adj]
        else:
            raise ValueError("shouldn't happen")

    # print("reached lower cases for p#{} i#{}".format(point_number, iteration))

    # children born this iteration are easy to find for points >= 2
    childL = get_child(point_number, 0, iteration)
    childDL = get_child(point_number, 1, iteration)
    childD = get_child(point_number, 2, iteration)
    neighbors = [childL, childDL, childD]
    if point_number < 12:
        neighbors += [None, None]
    else:
        neighbors += [None, None, None]
    
    # then the other two/three can be gotten as the children of the point's previous neighbors
    previous_adj = get_adjacency_recursive(point_number, iteration-1)
    assert len(neighbors) == len(previous_adj), "adjacency length for p#{} changed".format(point_number)
    for neigh_i in range(len(previous_adj)):
        if neighbors[neigh_i] is None:
            previous_neighbor = previous_adj[neigh_i]
            new_neighbor = get_generic_point_neighbor(previous_neighbor, point_number, iteration)  # the new neighbor of that point in the direction of this point
            neighbors[neigh_i] = new_neighbor

    return neighbors

    # --- old stuff; should no longer make any use of direction labels for this
    # previous_adj = unify_five_and_six(previous_adj, point_number)
    # neighborR = previous_adj[get_direction_number_from_label("R")]
    # neighborUR = previous_adj[get_direction_number_from_label("UR")]
    # neighborU = previous_adj[get_direction_number_from_label("U")]

    # neighborU_is_north_pole = neighborU == 0
    # neighborR_is_south_pole = neighborR == 1

    # # the new R is the previous R's L child (index 0)
    # childR = get_south_pole_neighbor(point_number, iteration) if neighborR_is_south_pole else get_child(neighborR, 0, iteration)
    # # the new UR is the previous UR's DL child (index 1)
    # childUR = get_child(neighborUR, 1, iteration)
    # # the new U is the previous U's D child (index 2)
    # childU = get_north_pole_neighbor(point_number, iteration) if neighborU_is_north_pole else get_child(neighborU, 2, iteration)

    # adj_labels = {
    #     "L": childL, "DL": childDL, "D": childD,
    #     "R": childR, "UR": childUR, "U": childU,
    # }
    # print("got adj_labels {}".format(adj_labels))
    # adj = convert_adjacency_label_dict_to_list(adj_labels)
    # return adj


def get_neighbor_clockwise_step(central_point, central_point_adjacency, reference_neighbor, n_steps):
    if central_point == 0:
        # clockwise around the north pole is DL, opposite of UR
        new_n_steps = -1 * n_steps
        return go_upright_from_north_pole_neighbor(reference_neighbor, central_point_adjacency, n_steps=new_n_steps)
    elif central_point == 1:
        # clockwise around the south pole is UR
        new_n_steps = n_steps
        return go_upright_from_south_pole_neighbor(reference_neighbor, central_point_adjacency, n_steps=new_n_steps)

    original_index = central_point_adjacency.index(reference_neighbor)
    # clockwise means going from L->DL->D->R->UR->U->L i.e. along the list toward the right, wrapping around
    new_index = (original_index + n_steps) % len(central_point_adjacency)  # this way we don't care whether parent has five neighbors or six; do you feel lucky, punk?
    return central_point_adjacency[new_index]


def get_index_clockwise_step(original_index, n_steps, n_neighbors):
    assert n_neighbors in [5, 6]
    return (original_index + n_steps) % n_neighbors


@functools.lru_cache(maxsize=10000)
def get_adjacency_when_born(point_number):
    # print("get_adjacency_when_born({})".format(point_number))
    iteration = get_iteration_born(point_number)

    if point_number < 12:
        point_list, adjacency_dict = STARTING_POINTS
        adj_raw = adjacency_dict[point_number]
        # print("returning known raw adjacency when born for starting point p#{}".format(point_number))
        return adj_raw  # just return the five-length one here in case this is the final call, only use the casting to six-length when it's an intermediate step to getting some non-initial point's adjacency

    # --- new stuff with attempts at unification

    parent = get_parent(point_number)
    child_index = get_child_index(point_number)
    parent_previous_adjacency = get_adjacency_recursive(parent, iteration-1)  # adjacency of parent in PREVIOUS iteration
    # the child index of this new point is the same as what its actual index in the adjacency list will be (of its parent)
    # since the adjacency list is in the order L,DL,D,R,UR,U
    # and children are made L,DL,D
    directional_parent = parent_previous_adjacency[child_index]
    parent_previous_cw_neighbor = get_neighbor_clockwise_step(parent, parent_previous_adjacency, directional_parent, 1)
    parent_previous_ccw_neighbor = get_neighbor_clockwise_step(parent, parent_previous_adjacency, directional_parent, -1)

    # shorthands
    point_a = parent_previous_cw_neighbor
    point_b = parent_previous_ccw_neighbor

    # the five-neighbor points are born from the aether at iteration 0, so we can assume this point has six neighbors
    neighbors = [None, None, None, None, None, None]
    # a parent's children will only ever be in its peel, and thus will share its child-direction orientation
    # thus, directional parent's direction index will be the same as this child's child index
    # and parent's index from child's perspective will be the opposite of that
    neighbors[child_index] = directional_parent
    index_of_parent = get_opposite_neighbor_direction(child_index)
    neighbors[index_of_parent] = parent

    # for the other four points, we know which edges they lie on, and they should(?) be the child of one or the other point around that edge
    
    # the point between point_a and parent
    # it is 1 counterclockwise from parent
    pa_par_index_from_child = get_index_clockwise_step(index_of_parent, n_steps=-1, n_neighbors=6)
    assert neighbors[pa_par_index_from_child] is None, "slot already filled"
    pa_par_from_a = get_generic_point_neighbor(point_a, parent, iteration)
    pa_par_from_par = get_generic_point_neighbor(parent, point_a, iteration)
    assert pa_par_from_a == pa_par_from_par
    pa_par = pa_par_from_a
    neighbors[pa_par_index_from_child] = pa_par

    # the point between point_a and directional_parent
    # it is 1 clockwise from dpar
    pa_dpar_index_from_child = get_index_clockwise_step(child_index, n_steps=1, n_neighbors=6)
    assert neighbors[pa_dpar_index_from_child] is None, "slot already filled"
    pa_dpar_from_a = get_generic_point_neighbor(point_a, directional_parent, iteration)
    pa_dpar_from_dpar = get_generic_point_neighbor(directional_parent, point_a, iteration)
    assert pa_dpar_from_a == pa_dpar_from_dpar
    pa_dpar = pa_dpar_from_a
    neighbors[pa_dpar_index_from_child] = pa_dpar

    # the point between point_b and parent
    # it is 1 clockwise from par
    pb_par_index_from_child = get_index_clockwise_step(index_of_parent, n_steps=1, n_neighbors=6)
    assert neighbors[pb_par_index_from_child] is None, "slot already filled"
    pb_par_from_b = get_generic_point_neighbor(point_b, parent, iteration)
    pb_par_from_par = get_generic_point_neighbor(parent, point_b, iteration)
    assert pb_par_from_b == pb_par_from_par
    pb_par = pb_par_from_b
    neighbors[pb_par_index_from_child] = pb_par

    # the point between point_b and directional_parent
    # it is 1 counterclockwise from dpar
    pb_dpar_index_from_child = get_index_clockwise_step(child_index, n_steps=-1, n_neighbors=6)
    assert neighbors[pb_dpar_index_from_child] is None, "slot already filled"
    pb_dpar_from_b = get_generic_point_neighbor(point_b, directional_parent, iteration)
    pb_dpar_from_dpar = get_generic_point_neighbor(directional_parent, point_b, iteration)
    assert pb_dpar_from_b == pb_dpar_from_dpar
    pb_dpar = pb_dpar_from_b
    neighbors[pb_dpar_index_from_child] = pb_dpar


    # --- old stuff with too many conditions, gonna try unifying it ---

    # see AdjacencyInduction.png about how to get the current adjacency from the previous generation's
    # parent = get_parent(point_number)
    # --- new stuff with attempts at unification
    # child_index = get_child_index(point_number)
    # parent_adjacency = get_adjacency_recursive(parent, iteration-1)  # adjacency of parent in PREVIOUS iteration
    # if parent < 12:
    #     assert parent not in [0, 1], "parent of p#{} is a pole ({}) but this should not happen".format(point_number, parent)
    #     # need to account that for the initial points other than the poles, there is a relation in which two of the neighbor directions point to the same neighbor point
    #     parent_adjacency = unify_five_and_six(parent_adjacency, parent)
    #     assert len(parent_adjacency) == 6, "parent p#{} has adjacency of wrong length: {}".format(parent, parent_adjacency)

    # print("using parent adjacency {}".format(parent_adjacency))
    # parL = parent_adjacency[get_direction_number_from_label("L")]
    # parDL = parent_adjacency[get_direction_number_from_label("DL")]
    # parD = parent_adjacency[get_direction_number_from_label("D")]
    # parR = parent_adjacency[get_direction_number_from_label("R")]
    # parUR = parent_adjacency[get_direction_number_from_label("UR")]
    # parU = parent_adjacency[get_direction_number_from_label("U")]

    # parL_is_north_pole = parL == 0
    # parU_is_north_pole = parU == 0
    # parR_is_south_pole = parR == 1
    # parD_is_south_pole = parD == 1
    # if parL_is_north_pole or parU_is_north_pole:
    #     previous_np_adj = get_adjacency_recursive(0, iteration-1)
    # if parR_is_south_pole or parD_is_south_pole:
    #     previous_sp_adj = get_adjacency_recursive(1, iteration-1)
    # parent_on_NR = is_initial_northern_ring_point(parent)
    # parent_on_SR = is_initial_southern_ring_point(parent)
    # on_northern_seam = is_on_northern_seam(point_number)
    # on_northern_seam_and_parent_has_five_neighbors = on_northern_seam and parent_on_NR
    # on_northern_seam_and_parent_has_six_neighbors = on_northern_seam and not parent_on_NR
    # on_southern_seam = is_on_southern_seam(point_number)
    # on_southern_seam_and_parent_has_five_neighbors = on_southern_seam and parent_on_SR
    # on_southern_seam_and_parent_has_six_neighbors = on_southern_seam and not parent_on_SR

    # so many conditions
    # ugggggh oh god why
    # if child_index == 0:
    #     # parent plus its L, DL, U neighbors will give you all six new neighbors
    #     neighbors = {}
    #     # the new point is between parent and L, so parent is the R neighbor
    #     neighbors["R"] = parent
    #     # the parent's old left neighbor is the new point's L neighbor
    #     neighbors["L"] = parL
    #     # the parent's upper neighbor's D child (index 2) is the new point's UR neighbor
    #     neighbors["UR"] = get_child(parUR, 1, iteration) if on_northern_seam_and_parent_has_five_neighbors else get_child(parU, 1, iteration) if on_northern_seam_and_parent_has_six_neighbors else get_child(parU, 2, iteration)
    #     # the parent's upper neighbor's DL child (index 1) is the new point's U neighbor
    #     neighbors["U"] = get_child(parUR, 0, iteration) if on_northern_seam_and_parent_has_five_neighbors else get_child(parU, 0, iteration) if on_northern_seam_and_parent_has_six_neighbors else get_child(parU, 1, iteration)
    #     # the parent's left neighbor's D child (index 2) is the new point's DL neighbor
    #     neighbors["DL"] = get_child(go_upright_from_north_pole_neighbor(parent, previous_np_adj, -1), 0, iteration) if parL_is_north_pole else get_child(parL, 2, iteration)
    #     # the parent's DL child (index 1) is the new point's D neighbor
    #     neighbors["D"] = get_child(parent, 1, iteration)
    # elif child_index == 1:
    #     # parent plus its L, DL, D neighbors will give you all six new neighbors
    #     neighbors = {}
    #     # the parent is the UR
    #     neighbors["UR"] = parent
    #     # the parent's DL is the DL
    #     neighbors["DL"] = parDL
    #     # the parent's D child (index 2) is the R
    #     neighbors["R"] = get_child(parent, 2, iteration)
    #     # the parent's L child (index 0) is the U
    #     neighbors["U"] = get_child(parent, 0, iteration)
    #     # the parent's left's D child (index 2) is the L
    #     neighbors["L"] = get_child(go_upright_from_north_pole_neighbor(parent, previous_np_adj, -1), 0, iteration) if parL_is_north_pole else get_child(parL, 2, iteration)
    #     # the parent's down's L child (index 0) is the D
    #     neighbors["D"] = get_child(go_upright_from_south_pole_neighbor(parent, previous_sp_adj, -1), 2, iteration) if parD_is_south_pole else get_child(parD, 0, iteration)
    # elif child_index == 2:
    #     # parent plus its DL, D, R neighbors will give you all six new neighbors
    #     neighbors = {}
    #     # the parent is the U
    #     neighbors["U"] = parent
    #     # the parent's D is the D
    #     neighbors["D"] = parD
    #     # the parent's R's L child (index 0) is the UR
    #     neighbors["UR"] = get_child(parUR, 1, iteration) if on_southern_seam_and_parent_has_five_neighbors else get_child(parR, 1, iteration) if on_southern_seam_and_parent_has_six_neighbors else get_child(parR, 0, iteration)
    #     # the parent's R's DL child (index 1) is the R
    #     neighbors["R"] = get_child(parUR, 2, iteration) if on_southern_seam_and_parent_has_five_neighbors else get_child(parR, 2, iteration) if on_southern_seam_and_parent_has_six_neighbors else get_child(parR, 1, iteration)
    #     # the parent's DL child (index 1) is the L
    #     neighbors["L"] = get_child(parent, 1, iteration)
    #     # the parent's D's L child (index 0) is the DL
    #     neighbors["DL"] = get_child(go_upright_from_south_pole_neighbor(parent, previous_sp_adj, -1), 2, iteration) if parD_is_south_pole else get_child(parD, 0, iteration)
    # else:
    #     raise RuntimeError("invalid child index encountered: {}".format(child_index))

    # print("got neighbors of p#{} when born at i={}: {}".format(point_number, iteration, neighbors))
    # return convert_adjacency_label_dict_to_list(neighbors)
    return neighbors  # try populating it as list by index rather than using direction labels at all


def is_on_northern_seam(p):
    # the seams are the edges touching the north pole, from the pole to its five original neighbors
    if is_initial_northern_ring_point(p):
        return True
    else:
        parent = get_parent(p)
        child_index = get_child_index(p)
        return child_index == 0 and is_on_northern_seam(parent)


def is_on_southern_seam(p):
    if is_initial_southern_ring_point(p):
        return True
    else:
        parent = get_parent(p)
        child_index = get_child_index(p)
        return child_index == 2 and is_on_southern_seam(parent)


def is_initial_northern_ring_point(p):
    return p in [2, 4, 6, 8, 10]


def is_initial_southern_ring_point(p):
    return p in [3, 5, 7, 9, 11]


def get_north_pole_neighbor(previous_neighbor_in_direction, iteration):
    if iteration < 1:
        raise ValueError("can't get north pole neighbor before iteration 1; use the initial adjacency for iteration 0")
    # the north pole's neighbor in this direction will be the left child of the previous neighbor
    return get_child(previous_neighbor_in_direction, child_index=0, iteration=iteration)


def get_south_pole_neighbor(previous_neighbor_in_direction, iteration):
    if iteration < 1:
        raise ValueError("can't get south pole neighbor before iteration 1; use the initial adjacency for iteration 0")
    # the south pole's neighbor in this direction will be the down child of the previous neighbor
    return get_child(previous_neighbor_in_direction, child_index=2, iteration=iteration)


def get_generic_point_neighbor(point, previous_neighbor_in_direction, iteration):
    # print("getting neighbor of p#{} at i={}, in the direction of neighbor p#{} at i={}".format(point, iteration, previous_neighbor_in_direction, iteration-1))
    if point == 0:
        return get_north_pole_neighbor(previous_neighbor_in_direction, iteration)
    elif point == 1:
        return get_south_pole_neighbor(previous_neighbor_in_direction, iteration)

    previous_neighbor_adjacency = get_adjacency_recursive(previous_neighbor_in_direction, iteration-1)
    previous_point_adjacency = get_adjacency_recursive(point, iteration-1)

    # we don't care if it's actually a parent and child, we just care that one of these directions is such that we can make a NEW child from one of the points (and thus get the child's number analytically)
    # print("checking child-directionality between {} and {}".format(point, previous_neighbor_in_direction))
    desired_point_is_in_child_direction_from_neighbor = is_parent_and_child_direction(previous_neighbor_in_direction, point, previous_neighbor_adjacency)
    desired_point_is_in_child_direction_from_point = is_parent_and_child_direction(point, previous_neighbor_in_direction, previous_point_adjacency)
    # print("done checking child-directionality, got {}>{} {}; {}>{} {}".format(previous_neighbor_in_direction, point, desired_point_is_in_child_direction_from_neighbor, point, previous_neighbor_in_direction, desired_point_is_in_child_direction_from_point))

    if desired_point_is_in_child_direction_from_neighbor:
        # get the neighbor's new child
        # child index should equal the adjacency index since adjacency is in order L,DL,D,R,UR,U and child index is in order L,DL,D
        index_of_point_from_neighbor_perspective = previous_neighbor_adjacency.index(point)
        child_index = index_of_point_from_neighbor_perspective
        return get_child(previous_neighbor_in_direction, child_index, iteration)
    elif desired_point_is_in_child_direction_from_point:
        index_of_neighbor_from_point_perspective = previous_point_adjacency.index(previous_neighbor_in_direction)
        child_index = index_of_neighbor_from_point_perspective
        return get_child(point, child_index, iteration)
    else:
        raise Exception("generic neighbor from {} to {} at i={} has no parent-child relation".format(point, previous_neighbor_in_direction, iteration-1))


def go_upright_from_north_pole_neighbor(starting_neighbor, adjacency, n_steps=1):
    # get the next neighbor of north pole to the east (up-right direction on peels)
    # the north pole's adjacency is in eastward direction
    assert len(adjacency) == 5
    original_index = adjacency.index(starting_neighbor)
    new_index = (original_index + n_steps) % 5
    return adjacency[new_index]


def go_upright_from_south_pole_neighbor(starting_neighbor, adjacency, n_steps=1):
    # get the next neighbor of south pole to the east (up-right direction on peels)
    # the south pole's adjacency is in WESTWARD (backward) direction!
    assert len(adjacency) == 5
    original_index = adjacency.index(starting_neighbor)
    new_index = (original_index - n_steps) % 5  # go backwards in the list to go east
    return adjacency[new_index]


def convert_adjacency_label_dict_to_list(neighbors):
    adj = []
    for neighbor_index in range(6):
        dir_label = get_direction_label_from_number(neighbor_index)
        adj.append(neighbors[dir_label])
    return adj


def get_child_index(point_number):
    # index that the point is for its parent in the generation it was born
    # if it was the Left child, 0; if it was the DownLeft child, 1; if it was the Down child, 2
    if point_number < 12:
        return None
    return point_number % 3


@functools.lru_cache(maxsize=10000)
def get_parent_positions(point_number):
    p0, p1 = get_parents(point_number)
    pos0 = get_position_recursive(p0)
    pos1 = get_position_recursive(p1)
    return pos0, pos1


def get_position_of_point_number_using_parents(point_number):
    if point_number < 12:
        pos, adj = STARTING_POINTS
        return pos[point_number].tuples
    pos0, pos1 = get_parent_positions(point_number)
    return get_position_of_child_from_parent_positions(pos0, pos1)


def get_position_of_child_from_parent_positions(pos0, pos1):
    p0 = UnitSpherePoint(pos0)
    p1 = UnitSpherePoint(pos1)
    midpoint = UnitSpherePoint.get_midpoint(p0, p1)
    xyz = midpoint.xyz()
    latlon = midpoint.latlondeg()
    return {"xyz": xyz, "latlondeg": latlon}


@functools.lru_cache(maxsize=10000)
def get_position_recursive(point_number):
    return get_position_of_point_number_using_parents(point_number)


def get_positions_recursive(point_numbers):
    # somehow need to make it efficient to do this for multiple points
    # e.g. they will probably run into same parents/grandparents/etc. at some point, those shouldn't be recalculated
    print(f"getting positions recursively for {len(point_numbers)} points")
    tree = get_ancestor_tree_for_multiple_points(point_numbers)
    print("got ancestor tree")
    pn_to_position = get_all_positions_in_ancestor_tree(tree)
    print(f"done getting positions recursively for {len(point_numbers)} points")
    return [pn_to_position[p] for p in point_numbers]

    # old, very slow
    # brute-force, just get each one individually
    # n_ps = len(point_numbers)
    # res = []
    # for i, pn in enumerate(point_numbers):
    #     if i % 1000 == 0:
    #         print(f"getting positions recursive; progress {i}/{n_ps}")
    #     pos = get_position_recursive(pn)
    #     res.append(pos)
    # return res


def get_3adder_for_iteration(i):
    # see IcosaParentChildRelations.ods for math
    # the 3adder is a function of iteration number, such that child_number = 3*(parent+adder)+child_index
    numer = (10 * (4 ** (i-1)) - 4)
    denom = 3
    assert numer % denom == 0, "iteration {} gave non-int 3adder: {}".format(i, res)
    return numer // denom  # avoid int() flooring for floats like x.9999


def get_iteration_born(point_number):
    if point_number < 0:
        raise ValueError("invalid point number {}".format(point_number))
    elif point_number < 12:
        return 0
    return math.ceil(get_exact_iterations_from_points(point_number+1))


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


def get_nearest_icosa_point_to_latlon(latlon, maximum_distance, planet_radius):
    lat, lon = latlon
    xyz = mcm.unit_vector_lat_lon_to_cartesian(lat, lon)
    return get_nearest_icosa_point_to_xyz(xyz, maximum_distance, planet_radius)


def get_nearest_icosa_point_to_xyz(xyz, maximum_distance, planet_radius):
    # print("getting nearest icosa point to {}".format(xyz))
    max_distance_normalized = maximum_distance / planet_radius
    candidate_usps, candidate_adjacencies = STARTING_POINTS
    iteration = 0
    while True:
        # print("i={}".format(iteration))
        nearest_candidate_usp, distance_normalized = get_nearest_neighbor_to_xyz(xyz, candidate_usps)
        assert nearest_candidate_usp.point_number is not None
        # print("nearest candidate is {} at distance of {}".format(nearest_candidate_usp, distance))
        if distance_normalized <= max_distance_normalized:
            # print("done getting nearest icosa point to {}".format(xyz))
            distance_in_units = distance_normalized * planet_radius
            return nearest_candidate_usp, distance_normalized, distance_in_units

        iteration += 1
        if iteration > 30:
            break
        nearest_candidate_neighbor_point_numbers = get_adjacency_recursive(nearest_candidate_usp.point_number, iteration)
        nearest_candidate_neighbors_usp = [get_usp_from_point_number(pi) for pi in nearest_candidate_neighbor_point_numbers]
        candidate_usps = nearest_candidate_neighbors_usp + [nearest_candidate_usp]

    raise RuntimeError("while loop ran too many times")


def get_nearest_neighbor_to_latlon(latlon, candidates_usp):
    lat, lon = latlon
    xyz = mcm.unit_vector_lat_lon_to_cartesian(lat, lon)
    candidates = [cand.xyz() for cand in candidates_usp]
    return get_nearest_neighbor_to_xyz(xyz, candidates_usp)


def get_nearest_neighbor_to_xyz(xyz, candidates_usp):
    min_distance = np.inf
    nearest_neighbors = []
    for c in candidates_usp:
        c_xyz = c.xyz()
        d = mcm.xyz_distance(xyz, c_xyz)
        if d < min_distance:
            nearest_neighbors = [c]
            min_distance = d
        elif d == min_distance:
            nearest_neighbors.append(c)
    if len(nearest_neighbors) == 1:
        return nearest_neighbors[0], min_distance
    else:
        raise RuntimeError("got more than one nearest neighbor to xyz {}: {}\nIf you are finding icosa points for an image lattice, try repositioning the image slightly so that it is not symmetric about the equator.".format(xyz, nearest_neighbors))


def get_usp_generator(iterations):
    print(f"getting usp generator for {iterations} iterations")
    n_points = get_points_from_iterations(iterations)
    for pi in range(n_points):
        usp = get_usp_from_point_number(pi)
        yield usp
    print(f"done getting usp generator for {iterations} iterations")


def get_xyz_generator(iterations):
    print(f"getting xyz generator for {iterations} iterations")
    n_points = get_points_from_iterations(iterations)
    for pi in range(n_points):
        xyz = get_xyz_from_point_number(pi)
        yield xyz
    print(f"done getting xyz generator for {iterations} iterations")


def get_latlon_generator(iterations):
    print(f"getting latlon generator for {iterations} iterations")
    n_points = get_points_from_iterations(iterations)
    for pi in range(n_points):
        latlon = get_latlon_from_point_number(pi)
        yield latlon
    print(f"getting latlon generator for {iterations} iterations")


def is_in_latlon_rectangle(lat, lon, min_lat, max_lat, min_lon, max_lon):
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def get_usps_in_latlon_rectangle(min_lat, max_lat, min_lon, max_lon, iterations):
    print(f"getting usps in latlon rectangle for {iterations} iterations. this function is very inefficient")
    g = get_usp_generator(iterations)
    res = []
    for p in g:
        lat, lon = p.latlondeg()
        if is_in_latlon_rectangle(lat, lon, min_lat, max_lat, min_lon, max_lon):
            res.append(p)
    print(f"done getting usps in latlon rectangle for {iterations} iterations")
    return res


def get_latlons_of_points_in_latlon_rectangle(min_lat, max_lat, min_lon, max_lon, iterations):
    print(f"getting latlons in latlon rectangle for {iterations} iterations. this function is very inefficient")
    # this will be horribly inefficient for large iterations and small rectangles since it's just brute force checking every point on the whole planet, so can optimize later if needed
    g = get_latlon_generator(iterations)
    res = []
    for lat, lon in g:
        if is_in_latlon_rectangle(lat, lon, min_lat, max_lat, min_lon, max_lon):
            res.append((lat, lon))
    print(f"done getting latlons in latlon rectangle for {iterations} iterations")
    return res


def notify_memo_accessed(memo_fp):
    # depending what I want at the time, maybe do nothing, maybe just print that it was accessed, or maybe raise exception if I'm trying to avoid any memoization at all
    # pass
    print("memo accessed: {}".format(memo_fp))
    # raise RuntimeError("memo accessed but shouldn't be: {}".format(memo_fp))  # advantage here that it will show the call stack


def test_parent_is_correct_neighbor():
    # in the iteration where a point is born, its parent must be to its R, UR, or U, depending which number child it is of that parent
    # in later iterations, the parent and child will be separated by intervening bisections of the edge connecting them
    for i in range(100):
        point_number = random.randint(12, 655362-1)
        n_iterations = get_iterations_needed_for_point_number(point_number)
        adj = get_specific_adjacency_from_memo(point_number, n_iterations)
        parent = get_parent(point_number)
        parent_point_direction_number = get_parent_point_direction_number(point_number)
        corresponding_neighbor = adj[parent_point_direction_number]
        assert parent == corresponding_neighbor
    print("test succeeded: parents are the correct neighbor nodes")


def test_children_are_correct_neighbors():
    # use function to calculate child number analytically = 3*(parent+adder)+child_index
    # verify that it matches the child numbers gotten from adjacency bisection
    for i in range(100):
        point_number = random.randint(12, 327682-1)  # exclude the last memoized iteration since they won't have children yet
        born_iteration = get_iteration_born(point_number)
        for n_iterations in range(born_iteration, 9):
            adj = get_specific_adjacency_from_memo(point_number, n_iterations)
            # print("p{} i{} adj: {}".format(point_number, n_iterations, adj))
            if n_iterations > born_iteration:
                children = [get_child(point_number, child_index, n_iterations) for child_index in [0,1,2]]
                # print("children", children)
                assert children == adj[:3]
    print("test succeeded: children are the correct neighbor nodes")


def test_adjacency_when_born():
    for i in range(162):
        # point_number = random.randint(0, 41)
        point_number = i
        print("checking adjacency when born of p#{}".format(point_number))
        res_no_memo = get_adjacency_when_born(point_number)
        res_memo = get_specific_adjacency_from_memo(point_number, get_iteration_born(point_number))
        assert res_no_memo == res_memo, "mismatch for p#{} when born:\ncomputed: {}\nmemoized: {}".format(point_number, res_no_memo, res_memo)
    print("test succeeded: adjacency calculated from scratch is the same as the memoized adjacency")


def test_adjacency_recursive(compare_memo=True):
    t0 = time.time()
    iterations_reported_as_having_no_memo = set()
    for i in range(1000):
        point = random.randint(0, 655362)
        born_iteration = get_iteration_born(point)
        iteration = max(born_iteration, random.randint(6, 20))
        adj = get_adjacency_recursive(point, iteration)
        print("\n-- test_adjacency_recursive p#{} i={}".format(point, iteration))
        print("adj: {}".format(adj))
        if compare_memo:
            try:
                adj_from_memo = get_specific_adjacency_from_memo(point, iteration)
                assert adj == adj_from_memo
            except FileNotFoundError:
                if iteration not in iterations_reported_as_having_no_memo:
                    print("skipping memo check for iteration {} because no memo file exists".format(iteration))
                    iterations_reported_as_having_no_memo.add(iteration)
                else:
                    pass
    if compare_memo:
        print("test succeeded: getting adjacency recursively matches memoized adjacency")
    else:
        print("test succeeded: finished computing adjacency recursively but did not check memo")
    t1 = time.time()
    print("time elapsed: {:.4f} seconds".format(t1-t0))


def test_pole_adjacency():
    for iteration in range(9):
        p0rec = get_adjacency_recursive(0, iteration)
        p0memo = get_specific_adjacency_from_memo(0, iteration)
        p1rec = get_adjacency_recursive(1, iteration)
        p1memo = get_specific_adjacency_from_memo(1, iteration)
        assert p0rec == p0memo, "north pole mismatch: {} vs {}".format(p0rec, p0memo)
        assert p1rec == p1memo, "north pole mismatch: {} vs {}".format(p1rec, p1memo)
    print("test succeeded: pole adjacency calculated from scratch is the same as the memoized adjacency")


def test_get_generic_point_neighbor():
    for point in range(0, 20):
        born_iteration = get_iteration_born(point)
        for di in [1, 2, 3]:
            iteration = born_iteration + di  # don't make it the born generation, so we'll have a previous neighbor
            for neigh_index in range(6 if point >= 12 else 5):
                print("\n-- test_get_generic_point_neighbor: case p#{} i={} ni={}".format(point, iteration, neigh_index))
                previous_neighbor_in_direction = get_specific_adjacency_from_memo(point, iteration-1)[neigh_index]
                new_p = get_generic_point_neighbor(point, previous_neighbor_in_direction, iteration)
                # print(new_p)
                adj = get_specific_adjacency_from_memo(point, iteration)
                # print(adj[neigh_index])
                assert new_p == adj[neigh_index]
    print("test succeeded: generic point neighbor works")


def test_report_cada_ii_iteration_requirements():
    radius = CADA_II_RADIUS_KM
    for edge_length in [1000, 100, 10, 1, 0.1, 0.01, 0.001]:
        print("edge length {} km on Cada II requires {} iterations".format(edge_length, get_iterations_needed_for_edge_length(edge_length, radius)))


def test_position_recursive(compare_memo=True):
    t0 = time.time()
    iterations_reported_as_having_no_memo = set()
    for i in range(1000):
        point = random.randint(0, 655362)
        born_iteration = get_iteration_born(point)
        iteration = born_iteration
        pos = get_position_recursive(point)
        print("\n-- test_position_recursive p#{} i={}".format(point, iteration))
        print("pos: {}".format(pos))
        if compare_memo:
            try:
                pos_from_memo = get_specific_position_from_memo(point)
                assert np.allclose(pos["xyz"], pos_from_memo["xyz"], rtol=1e-6), "{} does not match {}".format(pos, pos_from_memo)
                assert np.allclose(pos["latlondeg"], pos_from_memo["latlondeg"], rtol=1e-6), "{} does not match {}".format(pos, pos_from_memo)
            except FileNotFoundError:
                if iteration not in iterations_reported_as_having_no_memo:
                    print("skipping memo check for iteration {} because no memo file exists".format(iteration))
                    iterations_reported_as_having_no_memo.add(iteration)
                else:
                    pass
    if compare_memo:
        print("test succeeded: getting position recursively matches memoized position")
    else:
        print("test succeeded: finished computing position recursively but did not check memo")
    t1 = time.time()
    print("time elapsed: {:.4f} seconds".format(t1-t0))


def test_get_nearest_point_to_latlon():
    maximum_distance = 0.001
    max_point_number = -1
    planet_radius = CADA_II_RADIUS_KM
    for i in range(10000):
        latlon = UnitSpherePoint.get_random_unit_sphere_point().latlondeg()
        p, distance = get_nearest_icosa_point_to_latlon(latlon, maximum_distance, planet_radius)
        max_point_number = max(max_point_number, p.point_number)
        print("result: {} which is {} units away from {}".format(p, distance*planet_radius, latlon))
    max_iter = get_iteration_born(max_point_number)
    points_needed = get_points_from_iterations(max_iter)
    print("test succeeded: got sufficiently near icosa points for various latlons; largest point number encountered was {}, which requires {} iterations, having a total of {} points".format(max_point_number, max_iter, points_needed))


STARTING_POINTS = get_starting_points_immutable()  # since this is called way too many times otherwise, just initialize it as a global constant that can be accessed by further functions, e.g. base case for recursive adjacency algorithm
STARTING_POINTS_ORDERED, STARTING_POINTS_ADJACENCY = STARTING_POINTS
# is it a bad idea to define the global later than the functions?


if __name__ == "__main__":
    # n_points = get_exact_points_from_iterations(n_iterations)
    # plot_coordinate_patterns(n_iterations)
    # point_numbers = np.random.randint(0, n_points, (100,))
    # print(get_specific_positions(point_numbers, n_iterations))    
    # test_parent_is_correct_neighbor()
    # test_children_are_correct_neighbors()

    # for point_number in range(0, 162):
    #     p0 = get_parent(point_number)
    #     p1 = get_directional_parent(point_number)
    #     print("{} {} -> {}".format(p0, p1, point_number))
    # print(get_position_of_point_number_using_parents(point_number))

    # point_numbers = [random.randint(10**3,10**6) for i in range(100)]
    point_numbers = list(range(2562))
    print(point_numbers)
    ancestry = get_ancestor_tree_for_multiple_points(point_numbers)
    pn_to_position = get_all_positions_in_ancestor_tree(ancestry)
    for pn in point_numbers:
        p0, p1 = ancestry[pn]
        print(f"{pn}\t{p0}\t{p1}")  # for exporting to Excel
    directional_parents = [ancestry[pn][1] for pn in point_numbers]
    plt.scatter(point_numbers, directional_parents)
    plt.show()
    test_directional_parent_correctness(point_numbers)
    # g = get_ancestor_graph_from_ancestor_tree(tree)
    # nx.draw(g, with_labels=True)
    # plt.show()

    # print(get_parent_chain(point_number))
    # print(get_adjacency_recursive(point_number, get_iteration_born(point_number)))

    # test_pole_adjacency()
    # test_adjacency_when_born()
    # test_get_generic_point_neighbor()
    # test_adjacency_recursive(compare_memo=False)
    # test_report_cada_ii_iteration_requirements()
    # test_position_recursive(compare_memo=False)
    # test_get_nearest_point_to_latlon()
