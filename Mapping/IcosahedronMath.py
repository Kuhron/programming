# try to decrease the amount of memory used up by large lattices
# keep info about Icosa coordinates in files, look things up in tables, rather than keeping it all in RAM


import math
# import functools  # only use caching if you have to, really try to make things more efficient
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import MapCoordinateMath as mcm
from UnitSpherePoint import UnitSpherePoint
import BoxCornerMapping as bc
from PointCodeArithmetic import add_direction_to_point_code, normalize_peel, apply_peel_offset



EARTH_RADIUS_KM = 6371
CADA_II_RADIUS_FACTOR = 2.116
CADA_II_RADIUS_KM = CADA_II_RADIUS_FACTOR * EARTH_RADIUS_KM


# @functools.lru_cache(maxsize=100000)
def get_point_number_from_point_code(point_code):
    # base cases
    if point_code is None:
        return None
    try:
        return {pc:i for i,pc in enumerate("ABCDEFGHIJKL")}[point_code]
        # thought about doing "ABC...".index, but notice that "abc".index("bc") == 1
    except KeyError:
        pass

    original_ancestor = point_code[0]
    original_ancestor_number = get_point_number_from_point_code(original_ancestor)
    ancestor_number = original_ancestor_number
    for i, n in enumerate(point_code[1:]):
        iteration = i + 1
        if n == "0":
            # number stays the same because the parent isn't moving
            child_number = ancestor_number
        else:
            child_index = int(n) - 1
            child_number = get_child_from_point_number(ancestor_number, child_index, iteration)
        # print(f"new child number {child_number} for ancestor code {ancestor_number}")
        ancestor_number = child_number

    res = ancestor_number
    return res


# @functools.lru_cache(maxsize=100000)
def get_point_code_from_point_number(point_number):
    # base cases
    if point_number is None:
        return None
    try:
        return "ABCDEFGHIJKL"[point_number]
    except IndexError:
        pass

    res = ""
    parent_chain = get_parent_chain(point_number)
    for i, parent_dict in enumerate(parent_chain):
        this_ancestors_iteration_born = parent_dict["iteration born"]
        assert this_ancestors_iteration_born == i
        this_ancestors_parent_number = parent_dict["parent number"]
        this_ancestors_child_index = parent_dict["child index"]
        this_ancestors_point_number = parent_dict["point number"]
        if this_ancestors_iteration_born == 0:
            res = get_point_code_from_point_number(this_ancestors_point_number)
        else:
            assert len(res) == this_ancestors_iteration_born, "missed an ancestor somewhere"
            child_index_code = "0" if this_ancestors_child_index is None else str(this_ancestors_child_index + 1)
            res += child_index_code
    
    assert len(res) > 0, res
    return res


# @functools.lru_cache(maxsize=100000)
def get_latlon_from_point_code(point_code):
    xyz = get_xyz_from_point_code_recursive(point_code)
    return mcm.unit_vector_cartesian_to_lat_lon(*xyz)


# @functools.lru_cache(maxsize=100000)
def get_xyz_from_point_code(point_code):
    xyz = get_xyz_from_point_code_recursive(point_code)
    return xyz


# @functools.lru_cache(maxsize=100000)
def get_latlon_from_point_number(point_number):
    xyz = get_xyz_from_point_number_recursive(point_number)
    return mcm.unit_vector_cartesian_to_lat_lon(*xyz)


def get_latlons_from_point_numbers(point_numbers):
    return [get_latlon_from_point_number(pn) for pn in point_numbers]


def get_latlons_from_point_codes(point_codes):
    return [get_latlon_from_point_code(pc) for pc in point_codes]


# @functools.lru_cache(maxsize=100000)
def get_xyz_from_point_number(point_number):
    xyz = get_xyz_from_point_number_recursive(point_number)
    return xyz


def get_xyzs_from_point_numbers(point_numbers):
    return [get_xyz_from_point_number(pn) for pn in point_numbers]


def get_xyzs_from_point_codes(point_codes):
    return [get_xyz_from_point_code(pc) for pc in point_codes]


def get_xyz_array_from_point_numbers(point_numbers):
    xyzs = get_xyzs_from_point_numbers(point_numbers)
    arr = np.array(xyzs)
    assert arr.shape == (len(point_numbers), 3), arr.shape
    return arr


def get_usp_from_point_number(point_number):
    pos = get_xyz_from_point_number_recursive(point_number)
    return UnitSpherePoint.from_xyz(*pos, point_number=point_number)


def get_all_point_codes_in_order():
    # go through each iteration, keep track of how many points you have
    # don't re-yield the ones you already gave (and the duplicates should have trailing zeros)
    iteration = 0
    point_count = 0
    while True:
        g = get_all_point_codes_in_order_up_to_iteration_with_trailing_zeros(iteration)
        for i, code in enumerate(g):
            # print(f"got p#{i} code {code} from iteration {iteration}")
            if i < point_count:
                # if we've yielded the first 12,
                # point_count is 12 and i maxes at 11 for points to exclude
                assert code.endswith("0")
                continue  # don't re-yield
            else:
                assert not code.endswith("0")
                yield code
                point_count += 1
        iteration += 1


def get_all_point_codes_in_order_including_trailing_zero_repeats():
    iteration = 0
    while True:
        yield from get_all_point_codes_in_order_up_to_iteration_with_trailing_zeros(iteration)
        iteration += 1


def get_all_point_codes_in_order_up_to_iteration(iteration):
    # this one isn't recursive for building the point codes,
    # it just takes the output of the recursive one and strips the zeros off
    # to create valid point codes
    for pc in get_all_point_codes_in_order_up_to_iteration_with_trailing_zeros(iteration):
        pc = strip_trailing_zeros(pc)
        yield pc


def get_all_point_codes_in_order_up_to_iteration_with_trailing_zeros(iteration):
    if iteration == 0:
        for c in list("ABCDEFGHIJKL"):
            yield c
    else:
        previous_points = get_all_point_codes_in_order_up_to_iteration_with_trailing_zeros(iteration - 1)
        children_to_yield = []
        for i, code in enumerate(previous_points):
            # print(f"got point {code} from previous iteration {iteration - 1}")
            yield code + "0"  # do this no matter what so next call can use it to make children
            if i < 2:
                # ignore the poles since they can't reproduce, but still yield the pole itself
                continue
            else:
                # print(f"creating codes {code}[1,2,3]")
                children_to_yield += [code + x for x in list("123")]
        # only yield the children after all the points from the previous iteration have been yielded
        for code in children_to_yield:
            yield code


def get_all_point_codes_in_iteration(iteration):
    for pc in get_all_point_codes_in_order_up_to_iteration_with_trailing_zeros(iteration):
        pc_it = get_iteration_number_from_point_code(pc)
        if pc_it == iteration:
            yield pc


def get_random_point_code(min_iterations, expected_iterations, max_iterations):
    assert min_iterations <= expected_iterations <= max_iterations
    
    # start the string off with the minimum iterations needed
    s = random.choice("CDEFGHIJKL")
    for i in range(min_iterations - 1):
        s += random.choice("0123")
    backup_digit = random.choice("123")  # prevent trailing zeros from taking s back below min_iterations

    iterations_used = get_iteration_number_from_point_code(s)
    if iterations_used < expected_iterations:
        while True:
            s += random.choice("0123")
            iterations_used = get_iteration_number_from_point_code(s)
            if iterations_used >= max_iterations:
                break
            if random.random() < 1/(expected_iterations - min_iterations):
                break

    # based on the number of points in this iteration, maybe replace with one of the poles
    iterations_used = get_iteration_number_from_point_code(s)
    n_points_this_iteration = get_n_points_from_iterations(iterations_used)
    prob_of_pole = 2 / n_points_this_iteration
    if min_iterations == 0 and random.random() < prob_of_pole:
        return random.choice(["A", "B"])
    
    iteration_born = get_iteration_born_from_point_code(strip_trailing_zeros(s))
    if iteration_born < min_iterations:
        # replace the last zero with backup_digit
        s = s[:-1] + backup_digit
    else:
        s = strip_trailing_zeros(s)

    return s


def get_random_point_number(min_iterations, expected_iterations, max_iterations):
    pc = get_random_point_code(min_iterations, expected_iterations, max_iterations)
    pn = get_point_number_from_point_code(pc)
    return pn


def get_iteration_number_from_point_code(pc):
    # this INCLUDES trailing zeros, we want to know what iteration something is at
    # for iteration born, use get_iteration_born instead
    # the initial points have length 1, just their letter A-L
    return len(pc) - 1


def get_iterations_from_n_points(n):
    try:
        # n_points(n_iters) = 10 * (4**n_iters) + 2
        return {12: 0, 42: 1, 162: 2, 642: 3, 2562: 4, 10242: 5}[n]
    except KeyError:
        iterations = get_exact_iterations_from_n_points(n)
        assert iterations % 1 == 0, "number of points {} gave non-int number of iterations; make sure it is 2+10*(4**n)".format(n)
        return iterations


def get_n_points_from_iterations(n):
    try:
        return {0: 12, 1: 42, 2: 162, 3: 642, 4: 2562, 5: 10242}[n]
    except KeyError:
        points = get_exact_n_points_from_iterations(n)
        assert points % 1 == 0, "number of iterations {} gave non-int number of points {}".format(n, points)
        return points


def get_exact_n_points_from_iterations(n_iters):
    return 2 + 10 * (4 ** n_iters)


def get_exact_iterations_from_n_points(n_points):
    return np.log((n_points - 2)/10) / np.log(4)  # just use change of base since np.log(arr, b) doesn't like arrays


def get_iterations_needed_for_point_number(point_number):
    # n_points is at least point_number+1, e.g. if it's point number 12, you actually have 13 points
    n_points_min = point_number + 1
    if n_points_min <= 12:
        return 0  # the baseline icosa has 12 vertices
    iters_exact = get_exact_iterations_from_n_points(n_points_min)
    return math.ceil(iters_exact)


def verify_valid_point_numbers(point_numbers, n_iterations):
    points_at_iter = get_exact_n_points_from_iterations(n_iterations)
    for p in point_numbers:
        if type(p) is not int:
            raise TypeError("invalid point number, expected int: {}".format(p))
        if p < 0:
            raise ValueError("point number should be non-negative: {}".format(p))
        if p >= points_at_iter:
            raise ValueError("point number {} too high for {} iterations, which has {} points".format(p, n_iterations, points_at_iter))


def verify_can_have_children_from_point_number(pn, iteration):
    if pn in [0, 1]:
        raise ValueError(f"point {pn} cannot have children")
    # point cannot have children in the same iteration when it is born, so require that the point existed one iteration ago (or earlier)
    return iteration > get_iteration_born_from_point_number(pn)


def verify_can_have_children_from_point_code(pc, iteration):
    # iteration must be greater than point's born iteration
    if pc in ["A", "B"]:
        raise ValueError(f"point {pc} cannot have children")
    return iteration > get_iteration_born_from_point_code(pc)


def is_valid_point_number(point_number, n_iterations):
    try:
        verify_valid_point_numbers([point_number], n_iterations)
    except ValueError:
        return False
    return True


def can_have_children_from_point_number(point_number, n_iterations):
    try:
        verify_can_have_children_from_point_number(point_number, n_iterations)
    except ValueError:
        return False
    return True


def scatter_icosa_points_by_number(point_numbers, show=True):
    point_codes = [get_point_code_from_point_number(pn) for pn in point_numbers]
    scatter_icosa_points_by_code(point_codes, show=show)


def scatter_icosa_points_by_code(point_codes, show=True):
    latlons = get_latlons_from_point_codes(point_codes)
    lats = [ll[0] for ll in latlons]
    lons = [ll[1] for ll in latlons]
    plt.scatter(lats, lons)
    if show:
        plt.show()


def plot_neighbor_relationships(n_iterations):
    d = get_adjacency_memo_dict(n_iterations)
    n_points = get_exact_n_points_from_iterations(n_iterations)
    point_numbers = range(12, n_points)
    neighbor_indices = range(6)
    colors = ["red","yellow","green","blue","purple","black"]
    for ni, c in zip(neighbor_indices, colors):
        neighbor_numbers_at_index = [d[pi][ni] for pi in point_numbers]
        plt.scatter(point_numbers, neighbor_numbers_at_index, color=c, alpha=0.4)
    plt.show()


def plot_xyzs(n_iterations):
    d = get_position_memo_dict(n_iterations)
    n_points = get_exact_n_points_from_iterations(n_iterations)
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
    n_points = get_exact_n_points_from_iterations(n_iterations)
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


def get_starting_point_code_directional_dict():
    # this will help with finding the directional parent for a given point code
    # A and B don't have directions (L,DL,D = 1,2,3) coming from them
    return {
        "C": {"1": "A", "2": "K", "3": "L"},
        "D": {"1": "C", "2": "L", "3": "B"},
        "E": {"1": "A", "2": "C", "3": "D"},
        "F": {"1": "E", "2": "D", "3": "B"},
        "G": {"1": "A", "2": "E", "3": "F"},
        "H": {"1": "G", "2": "F", "3": "B"},
        "I": {"1": "A", "2": "G", "3": "H"},
        "J": {"1": "I", "2": "H", "3": "B"},
        "K": {"1": "A", "2": "I", "3": "J"},
        "L": {"1": "K", "2": "J", "3": "B"},
    }


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


# @functools.lru_cache(maxsize=10000)
def get_child_from_point_number(pn, child_index, iteration):
    if pn in [0, 1]:
        raise ValueError(f"point {pn} cannot have children")
    verify_can_have_children_from_point_number(pn, iteration)  # make sure parent exists and is old enough to have children
    adder = get_3adder_for_iteration(iteration)
    return 3 * (pn + adder) + child_index


def get_child_from_point_code(pc, child_index, iteration):
    if pc in ["A", "B"]:
        raise ValueError(f"point {pc} cannot have children")
    verify_can_have_children_from_point_code(pc, iteration)
    # pad with zeros if needed to get child at this iteration
    pc0 = pc.ljust(iteration, "0")
    child_index_str = str(child_index + 1)  # "0" is reserved for not moving and then making later child
    assert child_index_str in ["1", "2", "3"], child_index_str
    return pc0 + child_index_str


# @functools.lru_cache(maxsize=10000)
def get_children_from_point_number(pn, iteration):
    if pn in [0, 1]:
        raise ValueError("point {} cannot have children".format(pn))
    verify_can_have_children_from_point_number(pn, iteration)  # make sure parent exists and is old enough to have children
    adder = get_3adder_for_iteration(iteration)
    return [3 * (pn + adder) + child_index for child_index in [0, 1, 2]]


def get_parent_from_point_code(pc):
    if len(pc) == 1:
        return None
    if pc[-1] == "0":
        return pc[:-1]  # treat it as its own parent when it stays in the same place
    return strip_trailing_zeros(pc[:-1])


def get_directional_parent_from_point_code(pc):
    if pc[-1] == "0":
        return pc[:-1]  # treat it as its own dpar when it stays in the same place
    if len(pc) == 1:
        return None
    else:
        # convert it to the C-D peel and then convert the answer back to the correct peel
        cd_code, peel_offset = normalize_peel(pc)
        assert cd_code[0] in ["C", "D"], "peel normalization failed"

    cd_dpar = bc.get_directional_parent_from_point_code_using_box_corner_mapping(cd_code)
    
    # keep trailing zeros during the recursive calls to box corner mapping
    # but no longer need them here now that we have our final answer
    cd_dpar = strip_trailing_zeros(cd_dpar)

    # similarly undo reversed-polarity encoding if we got one of those
    if bc.point_code_is_in_reversed_polarity_encoding(cd_dpar):
        cd_dpar = bc.correct_reversed_edge_polarity(cd_dpar)

    dpar = apply_peel_offset(cd_dpar, peel_offset)
    return dpar


# @functools.lru_cache(maxsize=10000)
# def get_directional_parent_from_point_code_brute_force(point_code):
#     print("brute-forcing dpar from pc")
#     point_number = get_point_number_from_point_code(point_code)
#     dp_number = get_directional_parent_from_point_number(point_number)
#     dp_code = get_point_code_from_point_number(dp_number)
#     return dp_code


# @functools.lru_cache(maxsize=10000)
def get_parent_from_point_number(point_number):
    # each point except the initial 12 is created from a "parent", a pre-existing point from one of the previous iterations
    # at each iteration, each existing point except the poles gets three new children
    # so e.g. iteration 1 has 42 points, 40 of those get 3 children each, creating 120 new points, so iteration 2 has 162 points, correct
    # so each new generation skips 0 and 1 (the poles) and starts with point #2 in creating the children
    # e.g. from gen 1 to 2, start with point 2, create points 42,43,44, then 3>45,46,47, ..., 41>159,160,161
    if point_number < 12:
        return None  # initial points have no parents
    # see IcosaParentChildRelations.ods for math
    return point_number // 3 - get_3adder_for_iteration(get_iteration_born_from_point_number(point_number))


# @functools.lru_cache(maxsize=10000)
def get_directional_parent_from_point_number(point_number):
    # safe but slow: use known process of creation of new points, guarantees correct answer
    return get_directional_parent_via_inheritance(point_number)
    # risky but fast: use patterns seen in the numbers, but without proof that these generalizations will always hold
    # return get_directional_parent_via_numerology(point_number)


# @functools.lru_cache(maxsize=10000)
def get_directional_parent_via_inheritance(point_number):
    # the parent at the other end of the edge that was bisected to produce this point
    if point_number < 12:
        return None
    parent = get_parent_from_point_number(point_number)
    iteration_born = get_iteration_born_from_point_number(point_number)
    # print("{} was born i={}".format(point_number, iteration_born))
    parent_adjacency_in_born_iteration = get_adjacency_recursive(parent, iteration_born)
    parent_adjacency_in_previous_iteration = get_adjacency_recursive(parent, iteration_born-1)
    # print("parent previous adjacency: {}".format(parent_adjacency_in_previous_iteration))
    # print("parent current adjacency: {}".format(parent_adjacency_in_born_iteration))
    index_of_this_point_from_parent = parent_adjacency_in_born_iteration.index(point_number)
    previous_point_at_that_index = parent_adjacency_in_previous_iteration[index_of_this_point_from_parent]
    return previous_point_at_that_index
    # this avoids problems with indexing directions from the 12 initial points which only have 5 adjacencies with ill-defined directions


def test_directional_parent_correctness(point_numbers):
    failures = []
    for p in point_numbers:
        dp_inheritance = get_directional_parent_via_inheritance(p)
        # dp_numerology = get_directional_parent_via_numerology(p)
        raise
        if dp_inheritance != dp_numerology:
            failure_tup = (p, dp_inheritance, dp_numerology)
    if len(failures) == 0:
        print("success! directional parents all agreed")
    else:
        for p, dp_i, dp_n in failures:
            print(f"failure: point {p} has directional parent of {dp_i} by inheritance (which is the correct one), but numerology gave {dp_n}")


def get_parents_from_point_code(point_code):
    p0 = get_parent_from_point_code(point_code)
    p1 = get_directional_parent_from_point_code(point_code)
    return [p0, p1]


def get_parents_from_point_number(point_number):
    pc = get_point_code_from_point_number(point_number)
    p0_code = get_parent_from_point_code(pc)
    p1_code = get_directional_parent_from_point_code(pc)
    p0 = get_point_number_from_point_code(p0_code)
    p1 = get_point_number_from_point_code(p1_code)
    # print(f"#{point_number} = {pc} has parents #{p0} = {p0_code} and #{p1} = {p1_code}")
    return [p0, p1]


# @functools.lru_cache(maxsize=10000)
def get_parent_chain(point_number):
    # go forward in time, starting from the initial point which gives rise ultimately to this one
    # return list of tuples, one for each iteration starting with zero up to and including the one where this point was born
    # tuples are (iteration, parent, child_index, child) where parent is reproducing this generation and child is born this generation
    if point_number < 12:
        iteration_born = 0
        parent = None
        child_index = None
        child = point_number
        d = {
            "iteration born": iteration_born,
            "parent number": parent,
            "child index": child_index,
            "point number": child,
        }
        return [d]
    else:
        iteration_born = get_iteration_born_from_point_number(point_number)
        child_index = get_child_index_from_point_number(point_number)
        parent = get_parent_from_point_number(point_number)
        parent_iteration_born = get_iteration_born_from_point_number(parent)
        previous_chain = get_parent_chain(parent)
        chain = [x for x in previous_chain]
        iterations_with_same_parent = list(range(parent_iteration_born+1, iteration_born))
        for i in iterations_with_same_parent:
            this_child_index = None
            this_child = None
            d = {
                "iteration born": i,
                "parent number": parent,
                "child index": this_child_index,
                "point number": this_child,
            }
            chain.append(d)
        
        # dict for this point
        final_dict = {
            "iteration born": iteration_born,
            "parent number": parent,
            "child index": child_index,
            "point number": point_number,
        }
        chain.append(final_dict)
        assert len(chain) == iteration_born + 1
        return chain


def get_ancestor_tree(point_number, existing_ancestry=None):
    raise Exception("deprecated; can now efficiently calculate par and dpar on the fly")
    # # dict of child: parents
    # if existing_ancestry is None:
    #     existing_ancestry = {}
    # else:
    #     # for combining ancestry trees of multiple points, if we see a point's parent>child tuple already there,
    #     # then don't have to recalculate the rest of its ancestry which should also already be there
    #     pass

    # ancestry = {}
    # # use sets of tuples to take advantage of constant-time lookup
    # farthest_back_generation = [point_number]
    # while len(farthest_back_generation) > 0:
    #     new_farthest_back_generation = []
    #     for child in farthest_back_generation:
    #         if child in existing_ancestry or child in ancestry:
    #             # already know its parent
    #             continue
    #         parents = get_parents_from_point_number(child)
    #         parents = [x if x is not None else -1 for x in parents]  # convert to -1 for int sorting
    #         ancestry[child] = parents
    #         new_farthest_back_generation += parents
    #     farthest_back_generation = [x for x in new_farthest_back_generation if x != -1]
    # return ancestry


def get_ancestor_tree_for_multiple_points(point_numbers):
    raise Exception("deprecated; can now efficiently calculate par and dpar on the fly")
    # print(f"getting ancestor tree for {len(point_numbers)} points")
    # ancestry = {}
    # for i, p in enumerate(point_numbers):
    #     if i % 100 == 0:
    #         print(f"ancestor tree progress: {i}/{len(point_numbers)}")
    #     if p in ancestry:
    #         # don't need to call it
    #         continue
    #     else:
    #         p_ancestry = get_ancestor_tree(p, existing_ancestry=ancestry)
    #         ancestry.update(p_ancestry)
    #         assert p in ancestry, "failed to update ancestry correctly, should be adding parents of current point"
    # print(f"done getting ancestor tree for {len(point_numbers)} points")
    # return ancestry


def get_ancestor_graph(point_number):
    raise Exception("deprecated; can now efficiently calculate par and dpar on the fly")
    # ancestry = get_ancestor_tree(point_number)
    # return get_ancestor_graph_from_ancestor_tree(ancestry)


def get_ancestor_graph_for_multiple_points(point_numbers):
    raise Exception("deprecated; can now efficiently calculate par and dpar on the fly")
    # ancestry = get_ancestor_tree_for_multiple_points(point_numbers)
    # return get_ancestor_graph_from_ancestor_tree(ancestry)


def get_ancestor_graph_from_ancestor_tree(ancestry):
    raise Exception("deprecated; can now efficiently calculate par and dpar on the fly")
    # g = nx.DiGraph()
    # for child, (p0, p1) in ancestry.items():
    #     if p0 != -1:
    #         g.add_edge(p0, child)
    #     if p1 != -1:
    #         g.add_edge(p1, child)
    # return g


def get_all_positions_in_ancestor_tree(ancestry):
    raise Exception("deprecated; can now efficiently calculate par and dpar on the fly")
    # # auxiliary function meant to help make it easier to get the positions of a large number of points
    # # by taking advantage of the fact that many of them share ancestry, so that position info can be reused without recalculation
    # child_to_parents = ancestry
    # pn_to_position = {}
    # for child in sorted(child_to_parents.keys()):
    #     # do the lower-number points first because they are created earlier, and then later points can use position information from them
    #     p0, p1 = child_to_parents[child]
    #     # print(f"getting position from ancestry for child {child} of parents {p0}, {p1}")
    #     if p0 == -1 and p1 == -1:
    #         # print(f"getting position for parentless point {child}")
    #         pos = get_position_from_point_number_recursive(child)
    #         pn_to_position[child] = pos
    #     else:
    #         # print(f"getting position for point {child} with parents in the tree")
    #         pos0 = pn_to_position[p0]
    #         pos1 = pn_to_position[p1]
    #         # print(f"p0 is at {pos0}\np1 is at {pos1}")
    #         pos = get_position_of_child_from_parent_positions(pos0, pos1)
    #         # print(f"child is at {pos}")
    #         pn_to_position[child] = pos
    # return pn_to_position


def is_parent_and_child(parent, child):
    return parent == get_parent_from_point_number(child)


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
    assert type(adjacency) is list
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
        adj = adjacency[:3] + [adjacency[2],] + adjacency[3:]
    else:
        raise ValueError("invalid identity pair {}".format(identity_pair))

    return adj


def get_adjacency_from_point_number(pn, iteration, force_six_directions=False):
    return get_adjacency_recursive(pn, iteration, force_six_directions=force_six_directions)


def get_adjacency_from_point_code(pc, iteration, use_existing_method=False):
    # TODO get it more directly using the point code format
    # manipulating the symbols to easily tell where the parent is, etc.

    # par = get_parent_from_point_code(pc)
    # dpar = get_directional_parent_from_point_code(pc)

    if use_existing_method:
        # just copy existing logic for point numbers (works, but slow)
        pn = get_point_number_from_point_code(pc)
        pns = get_adjacency_from_point_number(pn, iteration, force_six_directions=True)
        adj = [get_point_code_from_point_number(pn1) if pn1 is not None else None for pn1 in pns]
        adj = [str(x).ljust(len(pc), "0") for x in adj]
        return adj

    neighL = add_direction_to_point_code(pc, 1)
    neighDL = add_direction_to_point_code(pc, 2)
    neighD = add_direction_to_point_code(pc, 3)
    neighR = add_direction_to_point_code(pc, -1)
    neighUR = add_direction_to_point_code(pc, -2)
    neighU = add_direction_to_point_code(pc, -3)
    return [neighL, neighDL, neighD, neighR, neighUR, neighU]


# @functools.lru_cache(maxsize=10000)
def get_adjacency_recursive(point_number, iteration, force_six_directions=False):
    # use get_adjacency_when_born() here as base case
    # for non-born iterations, use the formula for child number from parent, index, and iteration
    # print("getting adjacency recursive for p#{} in i#{}".format(point_number, iteration))

    if iteration == get_iteration_born_from_point_number(point_number):
        return get_adjacency_when_born_from_point_number(point_number, force_six_directions=force_six_directions)

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
    childL = get_child_from_point_number(point_number, 0, iteration)
    childDL = get_child_from_point_number(point_number, 1, iteration)
    childD = get_child_from_point_number(point_number, 2, iteration)
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


# @functools.lru_cache(maxsize=10000)
def get_adjacency_when_born_from_point_number(point_number, force_six_directions=False):
    # print("get_adjacency_when_born({})".format(point_number))
    iteration = get_iteration_born_from_point_number(point_number)

    # if point_number < 2:
    #     raise ValueError(f"directions from poles are ill-defined: {point_number=}")
    if point_number < 12:
        point_list, adjacency_dict = STARTING_POINTS
        adj_raw = adjacency_dict[point_number]
        adj_raw = list(adj_raw)
        if force_six_directions:
            if point_number in [0, 2, 4, 6, 8, 10]:  # northern ring
                adj_raw.append(None)  # -3 is undefined
            else:
                adj_raw = adj_raw[:3] + [None] + adj_raw[3:]  # -1 is undefined
        # print("returning known raw adjacency when born for starting point p#{}".format(point_number))
        return adj_raw  # just return the five-length one here in case this is the final call, only use the casting to six-length when it's an intermediate step to getting some non-initial point's adjacency

    # --- new stuff with attempts at unification

    parent = get_parent_from_point_number(point_number)
    child_index = get_child_index_from_point_number(point_number)
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

    # print("got neighbors of p#{} when born at i={}: {}".format(point_number, iteration, neighbors))
    return neighbors  # try populating it as list by index rather than using direction labels at all


def get_adjacency_when_born_from_point_code(pc):
    pc1 = strip_trailing_zeros(pc)
    return get_adjacency_from_point_code(pc1)


def is_on_northern_seam(p):
    # the seams are the edges touching the north pole, from the pole to its five original neighbors
    if is_initial_northern_ring_point(p):
        return True
    else:
        parent = get_parent_from_point_number(p)
        child_index = get_child_index_from_point_number(p)
        return child_index == 0 and is_on_northern_seam(parent)


def is_on_southern_seam(p):
    if is_initial_southern_ring_point(p):
        return True
    else:
        parent = get_parent_from_point_number(p)
        child_index = get_child_index_from_point_number(p)
        return child_index == 2 and is_on_southern_seam(parent)


def is_initial_northern_ring_point(p):
    return p in [2, 4, 6, 8, 10]


def is_initial_southern_ring_point(p):
    return p in [3, 5, 7, 9, 11]


def get_north_pole_neighbor(previous_neighbor_in_direction, iteration):
    if iteration < 1:
        raise ValueError("can't get north pole neighbor before iteration 1; use the initial adjacency for iteration 0")
    # the north pole's neighbor in this direction will be the left child of the previous neighbor
    return get_child_from_point_number(previous_neighbor_in_direction, child_index=0, iteration=iteration)


def get_south_pole_neighbor(previous_neighbor_in_direction, iteration):
    if iteration < 1:
        raise ValueError("can't get south pole neighbor before iteration 1; use the initial adjacency for iteration 0")
    # the south pole's neighbor in this direction will be the down child of the previous neighbor
    return get_child_from_point_number(previous_neighbor_in_direction, child_index=2, iteration=iteration)


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
        return get_child_from_point_number(previous_neighbor_in_direction, child_index, iteration)
    elif desired_point_is_in_child_direction_from_point:
        index_of_neighbor_from_point_perspective = previous_point_adjacency.index(previous_neighbor_in_direction)
        child_index = index_of_neighbor_from_point_perspective
        return get_child_from_point_number(point, child_index, iteration)
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


def get_child_index_from_point_number(pn):
    # index that the point is for its parent in the generation it was born
    # if it was the Left child, 0; if it was the DownLeft child, 1; if it was the Down child, 2
    if pn < 12:
        return None
    return pn % 3


def get_child_index_from_point_code(pc):
    s = pc[-1]
    return {"1": 0, "2": 1, "3": 2}[s]


# @functools.lru_cache(maxsize=10000)
def get_parent_xyzs_from_point_code(point_code):
    p0, p1 = get_parents_from_point_code(point_code)
    xyz0 = get_xyz_from_point_code_recursive(p0)
    xyz1 = get_xyz_from_point_code_recursive(p1)
    return xyz0, xyz1


# @functools.lru_cache(maxsize=10000)
def get_parent_xyzs_from_point_number(point_number):
    p0, p1 = get_parents_from_point_number(point_number)
    xyz0 = get_xyz_from_point_number_recursive(p0)
    xyz1 = get_xyz_from_point_number_recursive(p1)
    return xyz0, xyz1


def get_xyz_of_point_code_using_parents(point_code):
    originals = list("ABCDEFGHIJKL")
    if point_code in originals:
        pos, adj = STARTING_POINTS
        index = originals.index(point_code)
        # return pos[index].tuples
        return pos[index].xyz()
    xyz0, xyz1 = get_parent_xyzs_from_point_code(point_code)
    return get_xyz_of_child_from_parent_xyzs(xyz0, xyz1)


def get_xyz_of_point_number_using_parents(point_number):
    if point_number < 12:
        pos, adj = STARTING_POINTS
        # return pos[point_number].tuples
        return pos[point_number].xyz()
    xyz0, xyz1 = get_parent_xyzs_from_point_number(point_number)
    return get_xyz_of_child_from_parent_xyzs(xyz0, xyz1)


def get_xyz_of_child_from_parent_xyzs(xyz0, xyz1):
    # reduce use of UnitSpherePoint objects where they are unnecessary
    # also reduce use of latlon and conversion to/from it where it is unnecessary
    return mcm.get_unit_sphere_midpoint_from_xyz(xyz0, xyz1)


# @functools.lru_cache(maxsize=100000)
def get_xyz_from_point_code_recursive(point_code):
    return get_xyz_of_point_code_using_parents(point_code)


# @functools.lru_cache(maxsize=100000)
def get_xyz_from_point_number_recursive(point_number):
    return get_xyz_of_point_number_using_parents(point_number)


def get_xyzs_from_point_numbers_recursive(point_numbers):
    # somehow need to make it efficient to do this for multiple points
    # e.g. they will probably run into same parents/grandparents/etc. at some point, those shouldn't be recalculated

    return [get_xyz_from_point_number_recursive(pn) for pn in point_numbers]

    # old, slow
    # print(f"getting positions recursively for {len(point_numbers)} points")
    # tree = get_ancestor_tree_for_multiple_points(point_numbers)
    # print("got ancestor tree")
    # pn_to_position = get_all_positions_in_ancestor_tree(tree)
    # print(f"done getting positions recursively for {len(point_numbers)} points")
    # return [pn_to_position[p] for p in point_numbers]

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
    assert numer % denom == 0, "iteration {} gave non-int 3adder: {}".format(i, numer / denom)
    return numer // denom  # avoid int() flooring for floats like x.9999


def get_iteration_born_from_point_number(point_number):
    if point_number < 0:
        raise ValueError("invalid point number {}".format(point_number))
    elif point_number < 12:
        return 0
    return math.ceil(get_exact_iterations_from_n_points(point_number+1))


def get_iteration_born_from_point_code(pc):
    enforce_no_trailing_zeros(pc)
    return get_iteration_number_from_point_code(pc)


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
    print("getting nearest icosa point to {}".format(xyz))
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
    candidates_dict = {c.xyz(): c for c in candidates_usp}
    candidates_xyz = list(candidates_dict.keys())
    nn_xyz, d = get_nearest_neighbor_xyz_to_xyz(xyz, candidates_xyz)
    return candidates_dict[nn_xyz], d


def get_nearest_neighbor_to_xyz(xyz, candidates_usp):
    candidates_dict = {c.xyz(): c for c in candidates_usp}
    candidates_xyz = list(candidates_dict.keys())
    nn_xyz, d = get_nearest_neighbor_xyz_to_xyz(xyz, candidates_xyz)
    return candidates_dict[nn_xyz], d


def get_nearest_neighbor_point_number_to_point_number(pn, candidates_pn):
    xyz = get_xyz_from_point_number(pn)
    candidates_dict = {get_xyz_from_point_number(c): c for c in candidates_pn}
    candidates_xyz = list(candidates_dict.keys())
    nn_xyz, d = get_nearest_neighbor_xyz_to_xyz(xyz, candidates_xyz)
    return candidates_dict[nn_xyz], d


def get_nearest_neighbor_xyz_to_xyz(xyz, candidates_xyz):
    p = np.array(xyz)
    assert p.shape == (3,), p.shape
    ps = np.array(candidates_xyz)
    n = len(candidates_xyz)
    if n == 0:
        raise ValueError("empty candidates list")
    assert ps.shape == (n, 3), f"ps should have shape ({n}, 3) but got {ps.shape}:\n{ps}"
    dx = p-ps
    d = sum((dx**2).T) ** 0.5
    assert d.shape == (n,)
    nn_index = np.argmin(d)
    min_d = min(d)
    nn_arr = ps[nn_index]
    nn_xyz = tuple(nn_arr)
    return nn_xyz, min_d


def get_nearest_neighbors_pn_to_pn_with_distance(query_pns, candidate_pns, k_neighbors=1):
    pn_to_xyz = {}
    print("getting pn -> xyz mapping")
    all_pns = list(set(query_pns) | set(candidate_pns))
    for i, pn in enumerate(all_pns):
        if i % 100 == 0:
            print(f"pn -> xyz progress {i}/{len(all_pns)}")
        xyz = icm.get_xyz_from_point_number(pn)
        pn_to_xyz[pn] = xyz
    candidate_xyzs = [pn_to_xyz[pn] for pn in candidate_pns]
    query_xyzs = [pn_to_xyz[pn] for pn in query_pns]

    print("creating KDTree")
    kdtree = KDTree(candidate_xyzs)  # ensure order is same as the pn list
    print("-- done creating KDTree")
    distances, nn_indices = kdtree.query(query_xyzs, k=k_neighbors)
    print("done querying KDTree")
    nn_index_lookup = {pn_query: nn_indices[i] for i, pn_query in enumerate(query_pns)}
    nn_index_to_pn = {index: points_to_interpolate_from[index] for index in nn_indices}
    nn_pn_lookup = {pn_query: nn_index_to_pn[nn_index_lookup[pn_query]] for pn_query in query_pns}
    d_lookup = {pn_query: distances[i] for i, pn_query in enumerate(qury_pns)}
    return nn_pn_lookup, d_lookup


def _old_iterative_get_nearest_neighbor_xyz_to_xyz(xyz, candidates_xyz):
    min_distance = np.inf
    nearest_neighbors = []
    for c_xyz in candidates_xyz:
        d = mcm.xyz_distance(xyz, c_xyz)
        if d < min_distance:
            nearest_neighbors = [c_xyz]
            min_distance = d
        elif d == min_distance:
            nearest_neighbors.append(c_xyz)
    if len(nearest_neighbors) == 1:
        return nearest_neighbors[0], min_distance
    else:
        raise RuntimeError("got more than one nearest neighbor to xyz {}: {}\nIf you are finding icosa points for an image lattice, try repositioning the image slightly so that it is not symmetric about the equator.".format(xyz, nearest_neighbors))


def get_usp_generator(iterations):
    print(f"getting usp generator for {iterations} iterations")
    n_points = get_n_points_from_iterations(iterations)
    for pi in range(n_points):
        usp = get_usp_from_point_number(pi)
        yield usp
    print(f"done getting usp generator for {iterations} iterations")


def get_xyz_generator(iterations):
    print(f"getting xyz generator for {iterations} iterations")
    n_points = get_n_points_from_iterations(iterations)
    for pi in range(n_points):
        xyz = get_xyz_from_point_number(pi)
        yield xyz
    print(f"done getting xyz generator for {iterations} iterations")


def get_latlon_generator(iterations):
    print(f"getting latlon generator for {iterations} iterations")
    n_points = get_n_points_from_iterations(iterations)
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


def get_farthest_distance_descendant_can_be(point, radius=1, iteration_of_next_child=None):
    # when the point is born, it has a L, DL, and D neighbor, toward which any of its descendants will go, and the farthest that descent line can move is arbitrarily close to one of those three points
    # for purposes of filtering out points in a given region (distance from a given latlon) by stopping the traversal of ancestral lines that will always be too far away
    if point in [0, 1]:
        # these have no descendants
        return 0
    if iteration_of_next_child is None:
        iteration_checking_at = get_iteration_born_from_point_number(point)
    else:
        iteration_checking_at = iteration_of_next_child - 1
    adjacency_at_iter = get_adjacency_recursive(point, iteration_checking_at)
    if point < 12:
        adjacency_at_iter = unify_five_and_six(adjacency_at_iter, point)
    l, dl, d, _, _, _ = adjacency_at_iter
    xyz = get_xyz_from_point_number(point)
    distance_to_l = get_distance_icosa_point_to_xyz_great_circle(l, xyz)
    distance_to_dl = get_distance_icosa_point_to_xyz_great_circle(dl, xyz)
    distance_to_d = get_distance_icosa_point_to_xyz_great_circle(d, xyz)
    return radius * max(distance_to_l, distance_to_dl, distance_to_d)


def get_distance_icosa_point_to_xyz_great_circle(pn, xyz, radius=1):
    xyz2 = get_xyz_from_point_number(pn)
    return UnitSpherePoint.distance_great_circle_xyz_static(xyz, xyz2, radius=radius)


def get_distance_icosa_points_great_circle(pn1, pn2, radius=1):
    xyz1 = get_xyz_from_point_number(pn1)
    xyz2 = get_xyz_from_point_number(pn2)
    return UnitSpherePoint.distance_great_circle_xyz_static(xyz1, xyz2, radius=radius)


def print_pars_and_dpars_numbers_and_codes(iteration):
    n = get_n_points_from_iterations(iteration)
    for pn in range(n):
        pn0 = get_parent_from_point_number(pn)
        pn1 = get_directional_parent_from_point_number(pn)
        pc = get_point_code_from_point_number(pn)
        pc0 = get_point_code_from_point_number(pn0)
        pc1 = get_point_code_from_point_number(pn1)
        print(f"\tpn {pn}\tpc {pc}\nparent\tpn {pn0}\tpc {pc0}\ndirpar\tpn {pn1}\tpc {pc1}\n")


def get_dpar_dicts_up_to_iteration(iteration, first_dchildren_only=True):
    # in the point code system, since parents are obvious,
    # just care about how the directional parents work

    # all dpars can get more dchildren by taking any dchild and repeating its last digit
    # e.g. K1 has directional children K01, C12, C21, K011, C122, C211, K0111, C1222, etc.
    # the dchildren without repeating last digit are the "first directional children"

    points = get_all_point_codes_in_order()
    n = get_n_points_from_iterations(iteration)
    dpar_by_point = {}
    children_by_dpar = {}
    for i, pc in enumerate(points):
        if i % 1000 == 0:
            print(f"getting dpar dicts, point {i}/{n}")
        if i > n:
            raise RuntimeError("shouldn't happen")
        elif i == n:
            # reached end of points for this iteration
            break

        if has_repeating_last_digit(pc):
            continue
        # par = get_parent_from_point_code(pc)
        dpar = get_directional_parent_from_point_code(pc)
        dpar_by_point[pc] = dpar
        if dpar not in children_by_dpar:
            children_by_dpar[dpar] = []
        children_by_dpar[dpar].append(pc)
    return dpar_by_point, children_by_dpar


def has_repeating_last_digit(s):
    if len(s) < 2:
        return False
    # suffices only to check the last two
    # if there are more repetitions before that then this is still true
    return s[-1] == s[-2]


def strip_repeating_last_digits(s):
    while has_repeating_last_digit(s):
        s = s[:-1]
    return s


def strip_trailing_zeros(s):
    while len(s) > 0 and s[-1] == "0":
        s = s[:-1]
    return s


def enforce_no_trailing_zeros(pc):
    assert pc[-1] != "0", f"point code shouldn't have trailing zeros, got {pc}"


def print_first_dchildren(iteration):
    dpar_by_point, children_by_dpar = get_dpar_dicts_up_to_iteration(iteration, first_dchildren_only=True)
    for dpar, children in children_by_dpar.items():
        print(f"dpar {dpar} has directional children {children}")


def show_first_digit_dpar_relations(max_iteration):
    raise Exception("deprecated")
    # for helping me with pattern recognition for understanding how dpars work
    # one pattern is that the same first digit will be added to the dpar as was added to the point
    # e.g. H2102 has dpar of H212, while H-102 has dpar of H-12
    # but this isn't true in general, so see what points don't obey it

    # dpar_by_point, children_by_dpar = get_dpar_dicts_up_to_iteration(iteration=max_iteration, first_dchildren_only=True)
    # obeying_pattern = []
    # not_obeying_pattern = []
    # for pc, dpar in dpar_by_point.items():
    #     if len(pc) <= 2:
    #         # original point or iteration 1, ignore these since they're just memorized
    #         continue
    #     print("point\tdpar")
    #     print(f"{pc}\t{dpar}")
    #     i = 1
    #     # observe the dpar of the point gotten by removing this digit
    #     related_pc = pc[:i] + pc[i+1:]
    #     related_pc_display = pc[:i] + "-" + pc[i+1:]
    #     related_pc_key = strip_repeating_last_digits(related_pc)  # since dpar will be the same after stripping these
    #     related_pc_key = strip_trailing_zeros(related_pc_key)
    #     related_dpar = dpar_by_point[related_pc_key]
    #     related_dpar_display = related_dpar[0] + "-" + related_dpar[1:] if related_dpar is not None else None
    #     print(f"{related_pc_display}\t{related_dpar_display}")
    #     print()

    #     tup = (pc, dpar, related_pc_display, related_dpar_display)
    #     dpar_is_same_other_than_missing_digit = related_dpar_display == dpar[0] + "-" + dpar[2:]
    #     pc_is_same_other_than_missing_digit = related_pc_display == pc[0] + "-" + pc[2:]
    #     assert pc_is_same_other_than_missing_digit, "problem in how you coded this"
    #     missing_digit_is_same = pc[1] == dpar[1]
    #     obeys_pattern = dpar_is_same_other_than_missing_digit and pc_is_same_other_than_missing_digit and missing_digit_is_same
    #     if obeys_pattern:
    #         obeying_pattern.append(tup)
    #     else:
    #         not_obeying_pattern.append(tup)
    
    # for tup in obeying_pattern:
    #     print(f"obeys: {tup}")
    # print()
    # for tup in not_obeying_pattern:
    #     print(f"does not obey: {tup}")
    # print()


def plot_directional_parent_graph(iteration):
    g = nx.DiGraph()
    dpar_by_point, children_by_dpar = get_dpar_dicts_up_to_iteration(iteration, first_dchildren_only=True)
    for pc, dpar in dpar_by_point.items():
        if dpar is not None:
            g.add_edge(dpar, pc)
    nx.draw(g, with_labels=True)
    plt.show()


def get_region_around_point(point_number, iteration, max_distance_gc_normalized):
    # follow adjacency paths at this iteration resolution until you get every point within the radius
    verify_valid_point_numbers([point_number], iteration)
    res = {point_number}
    new_points = [point_number]
    while len(new_points) > 0:
        pn = new_points[0]
        neighbors = get_adjacency_from_point_number(pn, iteration)
        print(f"got neighbors of {pn}: {neighbors}")
        for neighbor in neighbors:
            if neighbor not in res:
                d = get_distance_icosa_points_great_circle(point_number, neighbor, radius=1)
                if d <= max_distance_gc_normalized:
                    new_points.append(neighbor)
        new_points = new_points[1:]
    return res


def test_parent_is_correct_neighbor():
    # in the iteration where a point is born, its parent must be to its R, UR, or U, depending which number child it is of that parent
    # in later iterations, the parent and child will be separated by intervening bisections of the edge connecting them
    for i in range(100):
        point_number = random.randint(12, 655362-1)
        n_iterations = get_iterations_needed_for_point_number(point_number)
        adj = get_adjacency_from_point_number(point_number, n_iterations)
        parent = get_parent_from_point_number(point_number)
        parent_point_direction_number = get_parent_point_direction_number(point_number)
        corresponding_neighbor = adj[parent_point_direction_number]
        assert parent == corresponding_neighbor
    print("test succeeded: parents are the correct neighbor nodes")


def test_children_are_correct_neighbors():
    # use function to calculate child number analytically = 3*(parent+adder)+child_index
    # verify that it matches the child numbers gotten from adjacency bisection
    for i in range(100):
        point_number = random.randint(12, 327682-1)  # exclude the last memoized iteration since they won't have children yet
        born_iteration = get_iteration_born_from_point_number(point_number)
        for n_iterations in range(born_iteration, 9):
            adj = get_adjacency_from_point_number(point_number, n_iterations)
            # print("p{} i{} adj: {}".format(point_number, n_iterations, adj))
            if n_iterations > born_iteration:
                children = [get_child_from_point_number(point_number, child_index, n_iterations) for child_index in [0,1,2]]
                # print("children", children)
                assert children == adj[:3]
    print("test succeeded: children are the correct neighbor nodes")


def test_adjacency_when_born():
    for i in range(162):
        point_number = i
        print("checking adjacency when born of p#{}".format(point_number))
        res_no_memo = get_adjacency_when_born_from_point_number(point_number)
    print("test succeeded: adjacency calculated from scratch")


def test_adjacency():
    t0 = time.time()
    for i in range(1000):
        pn = random.randint(0, 655362)
        born_iteration = get_iteration_born_from_point_number(pn)
        iteration = max(born_iteration, random.randint(6, 20))
        adj = get_adjacency_from_point_number(pn, iteration)
        print("\n-- test_adjacency p#{} i={}".format(pn, iteration))
        print("adj: {}".format(adj))
    else:
        print("test succeeded: finished computing adjacency")
    t1 = time.time()
    print("time elapsed: {:.4f} seconds".format(t1-t0))


def test_get_generic_point_neighbor():
    for point in range(0, 20):
        born_iteration = get_iteration_born_from_point_number(point)
        for di in [1, 2, 3]:
            iteration = born_iteration + di  # don't make it the born generation, so we'll have a previous neighbor
            for neigh_index in range(6 if point >= 12 else 5):
                print("\n-- test_get_generic_point_neighbor: case p#{} i={} ni={}".format(point, iteration, neigh_index))
                previous_neighbor_in_direction = get_adjacency_from_point_number(point, iteration-1)[neigh_index]
                new_p = get_generic_point_neighbor(point, previous_neighbor_in_direction, iteration)
                # print(new_p)
                adj = get_adjacency_from_point_number(point, iteration)
                # print(adj[neigh_index])
                assert new_p == adj[neigh_index]
    print("test succeeded: generic point neighbor works")


def test_report_cada_ii_iteration_requirements():
    radius = CADA_II_RADIUS_KM
    for edge_length in [1000, 100, 10, 1, 0.1, 0.01, 0.001]:
        print("edge length {} km on Cada II requires {} iterations".format(edge_length, get_iterations_needed_for_edge_length(edge_length, radius)))


def test_get_nearest_point_to_latlon():
    maximum_distance = 0.001
    max_point_number = -1
    planet_radius = CADA_II_RADIUS_KM
    for i in range(10000):
        latlon = UnitSpherePoint.get_random_unit_sphere_point().latlondeg()
        p, distance = get_nearest_icosa_point_to_latlon(latlon, maximum_distance, planet_radius)
        max_point_number = max(max_point_number, p.point_number)
        print("result: {} which is {} units away from {}".format(p, distance*planet_radius, latlon))
    max_iter = get_iteration_born_from_point_number(max_point_number)
    points_needed = get_n_points_from_iterations(max_iter)
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

    # print_pars_and_dpars_numbers_and_codes(iteration=3)
    # point_code = get_point_code_from_point_number(point_number)
    # latlon = get_latlon_from_point_code(point_code)
    # print(f"# {point_number} = {point_code}, at {latlon}")
    # print(get_position_of_point_number_using_parents(point_number))
    # plot_directional_parent_graph(iteration=5)
    # print_first_dchildren(iteration=5)
    # show_first_digit_dpar_relations(max_iteration=4)
    # dpar_by_point, children_by_dpar = get_dpar_dicts_up_to_iteration(iteration=6, first_dchildren_only=True)

    # speed test
    # iterations = 8
    # n_points = get_n_points_from_iterations(iterations)
    # for i, pc in enumerate(get_all_point_codes_in_order_up_to_iteration(iterations)):
    # for i, pc in enumerate(get_all_point_codes_in_order()):
    # for pc in get_all_point_codes_in_iteration(1):
    i = 2
    while True:
        # in the point database as of 2022-07-15, there are 1625217 points with average iterations around 13
        # pc = get_random_point_code(min_iterations=3, expected_iterations=3, max_iterations=3)
        # pc = random.choice(["C", "D"]) + pc[1:]
        # pc = ("C" + "".join(random.choice("01") for i in range(3))) if random.random() < 0.5 else ("D" + "".join(random.choice("03") for i in range(3)))
        pc = get_point_code_from_point_number(i).ljust(5, "0")
        par = get_parent_from_point_code(pc)
        dpar = get_directional_parent_from_point_code(pc)
        # latlon = get_latlon_from_point_code(pc)
        iteration = get_iteration_number_from_point_code(pc)
        adj_test = get_adjacency_from_point_code(pc, iteration, use_existing_method=False)
        adj_known = get_adjacency_from_point_code(pc, iteration, use_existing_method=True)
        # print(f"i={i}, {pc} <- {dpar}, point is located at {latlon}")
        print("p\tpar\tdpar\t+1\t+2\t+3\t-1\t-2\t-3")
        for adj in [adj_known, adj_test]:
            adj_str = "\t".join(str(x) for x in adj)
            print(f"{pc}\t{par}\t{dpar}\t{adj_str}")
        for neigh_known, neigh_test in zip(adj_known, adj_test):
            if neigh_test not in [None, "?"] and neigh_known != neigh_test:
                raise RuntimeError(f"Warning: {neigh_test=}, {neigh_known=}")
        print("----")
        i += 1

    raise
    point_number = get_random_point_number(min_iterations=6, expected_iterations=8)
    iteration = get_iteration_born_from_point_number(point_number)
    region = get_region_around_point(point_number, iteration, max_distance_gc_normalized=0.01)
    print(region)
    scatter_icosa_points_by_number(region, show=True)

    # point_numbers = [random.randint(10**3,10**6) for i in range(100)]
    # point_numbers = list(range(2562))
    # print(point_numbers)
    # ancestry = get_ancestor_tree_for_multiple_points(point_numbers)
    # pn_to_position = get_all_positions_in_ancestor_tree(ancestry)
    # for pn in point_numbers:
    #     p0, p1 = ancestry[pn]
    #     print(f"{pn}\t{p0}\t{p1}")  # for exporting to Excel
    # directional_parents = [ancestry[pn][1] for pn in point_numbers]
    # plt.scatter(point_numbers, directional_parents)
    # plt.show()
    # test_directional_parent_correctness(point_numbers)
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
