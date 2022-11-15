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
def get_point_code_from_point_number(pn):
    # base cases
    if pn is None:
        return None
    if pn < 12:
        return "ABCDEFGHIJKL"[pn]

    digit_dict = {}
    current_number = pn
    while current_number >= 12:
        child_index = get_child_index_from_point_number(current_number)
        iteration = get_iteration_born_from_point_number(current_number)
        digit_dict[iteration] = str(child_index + 1)
        current_number = get_parent_from_point_number(current_number)
    pc = "ABCDEFGHIJKL"[current_number]
    iteration_born = get_iteration_born_from_point_number(pn)
    for i in range(1, iteration_born + 1):
        pc += digit_dict.get(i, "0")
    res = pc
    assert len(res) > 0, res
    return res


def get_latlon_from_point_code(pc):
    if len(pc) == 1:
        return get_latlon_of_initial_point_code(pc)
    else:
        xyz = get_xyz_from_point_code_recursive(pc)
        return mcm.unit_vector_cartesian_to_lat_lon(*xyz)


def get_xyz_from_point_code(pc):
    if len(pc) == 1:
        pn = get_point_number_from_point_code(pc)
        return get_xyz_from_point_number(pn)
    else:
        xyz = get_xyz_from_point_code_recursive(pc)
        return xyz


def get_latlon_of_initial_point_number(pn):
    pos, adj = STARTING_POINTS
    return pos[pn].latlondeg()


def get_latlon_of_initial_point_code(pc):
    pn = get_point_number_from_point_code(pc)
    return get_latlon_of_initial_point_number(pn)


def get_latlon_from_point_number(pn):
    xyz = get_xyz_from_point_number_recursive(pn)
    return mcm.unit_vector_cartesian_to_lat_lon(*xyz)


def get_latlons_from_point_numbers(pns):
    return [get_latlon_from_point_number(pn) for pn in pns]


def get_latlons_from_point_codes(pcs):
    return [get_latlon_from_point_code(pc) for pc in pcs]


def get_xyz_from_point_number(pn):
    if pn < 12:
        return get_xyz_of_initial_point_number(pn)
    xyz = get_xyz_from_point_number_recursive(pn)
    return xyz


def get_xyzs_from_point_numbers(pns):
    return [get_xyz_from_point_number(pn) for pn in pns]


def get_xyzs_from_point_codes(pcs):
    return [get_xyz_from_point_code(pc) for pc in pcs]


def get_xyz_array_from_point_numbers(pns):
    xyzs = get_xyzs_from_point_numbers(pns)
    arr = np.array(xyzs)
    assert arr.shape == (len(pns), 3), arr.shape
    return arr


def get_usp_from_point_number(pn):
    pos = get_xyz_from_point_number_recursive(pn)
    return UnitSpherePoint.from_xyz(*pos, point_number=pn)


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
    assert type(n) is int
    return get_exact_n_points_from_iterations(n)


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


def verify_valid_point_numbers(pns, n_iterations):
    points_at_iter = get_exact_n_points_from_iterations(n_iterations)
    for pn in pns:
        if type(pn) is not int:
            raise TypeError("invalid point number, expected int: {}".format(pn))
        if pn < 0:
            raise ValueError("point number should be non-negative: {}".format(pn))
        if pn >= points_at_iter:
            raise ValueError("point number {} too high for {} iterations, which has {} points".format(pn, n_iterations, points_at_iter))


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


def scatter_icosa_points_by_code(point_codes, show=True, **kwargs):
    latlons = get_latlons_from_point_codes(point_codes)
    lats = [ll[0] for ll in latlons]
    lons = [ll[1] for ll in latlons]
    plt.scatter(lons, lats, **kwargs)
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


def get_children_from_point_code(pc):
    # including the null child where we keep the point in the same place
    if pc[0] in ["A", "B"]:
        # pole can't have children
        if all(y == "0" for y in pc[1:]):
            # it's actually the pole
            # but we need to return the pole itself at this iteration (pseudo-child)
            return [pc + "0"]
        else:
            raise ValueError(f"got reverse-encoded or invalid point code {pc}")
    else:
        return [pc + x for x in "0123"]


def get_parent_from_point_code(pc):
    if len(pc) == 1:
        return None
    if pc[-1] == "0":
        return pc[:-1]  # treat it as its own parent when it stays in the same place
    return strip_trailing_zeros(pc[:-1])


def get_directional_parent_from_point_number(pn):
    pc = get_point_code_from_point_number(pn)
    dpar_code = get_directional_parent_from_point_code(pc)
    return get_point_number_from_point_code(dpar_code)


def get_directional_parent_from_point_code(pc):
    if pc[-1] == "0":
        return pc[:-1]  # treat it as its own dpar when it stays in the same place
    if len(pc) == 1:
        return None
    
    reference_peel = bc.get_peel_containing_point_code(pc)

    if bc.point_code_is_in_reversed_polarity_encoding(pc):
        raise ValueError(f"got reverse-encoded {pc=}, please correct its encoding somewhere that we know which peel's perspective we are in")
    
    # faster than using adjacency and getting dpar at same index as this point's child_index
    # convert it to the C-D peel and then convert the answer back to the correct peel
    pc, peel_offset = normalize_peel(pc)
    assert pc[0] in ["C", "D"], "peel normalization failed"

    # cd_dpar = bc.get_directional_parent_from_point_code_using_box_corner_mapping(pc)
    child_index = get_child_index_from_point_code(pc)
    adj = get_adjacency_from_point_code(pc)
    cd_dpar = adj[child_index]
    
    # keep trailing zeros during the recursive calls to box corner mapping
    # but no longer need them here now that we have our final answer
    cd_dpar = strip_trailing_zeros(cd_dpar)

    # similarly undo reversed-polarity encoding if we got one of those
    if bc.point_code_is_in_reversed_polarity_encoding(cd_dpar):
        rev_cd_dpar = bc.correct_reversed_edge_polarity(cd_dpar, reference_peel)
        # print(f"reversed {cd_dpar} to {rev_cd_dpar}")
        cd_dpar = rev_cd_dpar

    dpar = apply_peel_offset(cd_dpar, peel_offset)
    # print(f"offset {cd_dpar} to {dpar}")
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


def get_parents_from_point_code(pc):
    if len(pc) == 1:
        par = None
        dpar = None
    elif pc[-1] == "0":
        par = pc[:-1]
        dpar = pc[:-1]
    else:
        adj = get_adjacency_from_point_code(pc)
        ci = get_child_index_from_point_code(pc)
        par = adj[3+ci]
        dpar = adj[ci]
        assert par[-1] == "0", par
        assert dpar[-1] == "0", dpar
        par = par[:-1]
        dpar = dpar[:-1]
    # print(f"parents of {pc} are {par=}, {dpar=}")
    return [par, dpar]

    # old way
    # p0 = get_parent_from_point_code(pc)
    # p1 = get_directional_parent_from_point_code(pc)
    # return [p0, p1]


def get_parents_from_point_number(point_number):
    pc = get_point_code_from_point_number(point_number)
    return get_parent_from_point_code(pc)


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


def get_unordered_neighbors_from_point_number(pn, iteration):
    if pn in [0, 1]:
        pcs = get_unordered_neighbors_from_point_code(["A", "B"][pn], iteration)
        return set(get_point_number_from_point_code(pc) for pc in pcs)
    else:
        return set(get_adjacency_from_point_number(pn, iteration))


def get_unordered_neighbors_from_point_code(pc):
    iteration = get_iteration_number_from_point_code(pc)
    if pc[0] == "A":
        # just do this manually, it's only one special case type
        assert all(x == "0" for x in pc[1:]), f"invalid point code {pc}"
        return {p + "1" * iteration for p in "CEGIK"}
    elif pc[0] == "B":
        assert all(x == "0" for x in pc[1:]), f"invalid point code {pc}"
        return {p + "3" * iteration for p in "DFHJL"}
    else:
        return set(get_adjacency_from_point_code(pc))


def get_adjacency_from_point_number(pn, iteration=None):
    if iteration is None:
        iteration = get_iteration_born_from_point_number(pn)
    pc = get_point_code_from_point_number(pn)
    # pad it to the desired iteration precision
    pc = pc.ljust(iteration + 1, "0")
    pcs = get_adjacency_from_point_code(pc)
    return [get_point_number_from_point_code(pc1) for pc1 in pcs]


def get_adjacency_from_point_code(pc):
    neighL = add_direction_to_point_code(pc, 1)
    neighDL = add_direction_to_point_code(pc, 2)
    neighD = add_direction_to_point_code(pc, 3)
    neighR = add_direction_to_point_code(pc, -1)
    neighUR = add_direction_to_point_code(pc, -2)
    neighU = add_direction_to_point_code(pc, -3)
    adj = [neighL, neighDL, neighD, neighR, neighUR, neighU]
    # print(f"{pc=} has {adj=}")
    return adj


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


def get_parent_xyzs_from_point_code(pc):
    # print(f"getting parent xyzs for {pc=}")
    p0, p1 = get_parents_from_point_code(pc)
    xyz0 = get_xyz_from_point_code(p0)
    xyz1 = get_xyz_from_point_code(p1)
    # print(f"\n{pc=} has parents {p0} with {xyz0=} and {p1} with {xyz1=}")
    return xyz0, xyz1


def get_parent_xyzs_from_point_number(pn):
    assert pn >= 12, "should get latlon/xyz of initial point directly"
    p0, p1 = get_parents_from_point_number(pn)
    xyz0 = get_xyz_from_point_number(p0)
    xyz1 = get_xyz_from_point_number(p1)
    return xyz0, xyz1


def get_xyz_of_initial_point_number(pn):
    assert pn < 12, pn
    pos, adj = STARTING_POINTS
    return pos[pn].xyz()


def get_xyz_of_initial_point_code(pc):
    pn = get_point_number_from_point_code(pc)
    return get_xyz_of_initial_point_number(pn)


def get_xyz_of_point_code_using_parents(pc):
    assert len(pc) > 1, "should get latlon/xyz of initial point directly"
    xyz0, xyz1 = get_parent_xyzs_from_point_code(pc)
    return get_xyz_of_child_from_parent_xyzs(xyz0, xyz1)


def get_xyz_of_point_number_using_parents(pn):
    assert pn >= 12, "should get latlon/xyz of initial point directly"
    xyz0, xyz1 = get_parent_xyzs_from_point_number(pn)
    return get_xyz_of_child_from_parent_xyzs(xyz0, xyz1)


def get_xyz_of_child_from_parent_xyzs(xyz0, xyz1):
    # reduce use of UnitSpherePoint objects where they are unnecessary
    # also reduce use of latlon and conversion to/from it where it is unnecessary
    res = mcm.get_unit_sphere_midpoint_from_xyz(xyz0, xyz1)
    # print(f"midpoint of\n{xyz0=} and\n{xyz1} is\n{res}")
    return res


# @functools.lru_cache(maxsize=100000)
def get_xyz_from_point_code_recursive(pc):
    assert len(pc) > 1, "should get latlon/xyz of initial point directly"
    return get_xyz_of_point_code_using_parents(pc)


# @functools.lru_cache(maxsize=100000)
def get_xyz_from_point_number_recursive(pn):
    assert pn >= 12, "should get latlon/xyz of initial point directly"
    return get_xyz_of_point_number_using_parents(pn)


def get_xyzs_from_point_numbers_recursive(pns):
    # somehow need to make it efficient to do this for multiple points
    # e.g. they will probably run into same parents/grandparents/etc. at some point, those shouldn't be recalculated

    return [get_xyz_from_point_number_recursive(pn) for pn in pns]

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


def get_iteration_born_from_point_number(pn):
    if pn < 0:
        raise ValueError("invalid point number {}".format(pn))
    elif pn < 12:
        return 0
    # stupid way that is hopefully faster
    i = 1
    while True:
        # e.g. n_points is 12 in iteration 0, so point number < 12 means it's in iteration 0
        n = get_n_points_from_iterations(i)
        if pn < n:
            return i
        i += 1

    # old way, np.log call is relatively slow for our purposes here
    # return math.ceil(get_exact_iterations_from_n_points(point_number+1))


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
        print("nearest candidate is {} at distance of {}".format(nearest_candidate_usp, distance_normalized))
        if distance_normalized <= max_distance_normalized:
            print("done getting nearest icosa point to {}".format(xyz))
            distance_in_units = distance_normalized * planet_radius
            return nearest_candidate_usp, distance_normalized, distance_in_units

        iteration += 1
        if iteration > 30:
            raise RuntimeError("while loop ran too many times")
        nearest_candidate_neighbor_point_numbers = get_adjacency_from_point_number(nearest_candidate_usp.point_number, iteration)
        nearest_candidate_neighbor_point_numbers = [x for x in nearest_candidate_neighbor_point_numbers if x is not None]
        nearest_candidate_neighbors_usp = [get_usp_from_point_number(pi) for pi in nearest_candidate_neighbor_point_numbers]
        candidate_usps = nearest_candidate_neighbors_usp + [nearest_candidate_usp]


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
        xyz = get_xyz_from_point_number(pn)
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


def get_farthest_distance_descendant_can_be(pn, radius=1, iteration_of_next_child=None):
    # when the point is born, it has a L, DL, and D neighbor, toward which any of its descendants will go, and the farthest that descent line can move is arbitrarily close to one of those three points
    # for purposes of filtering out points in a given region (distance from a given latlon) by stopping the traversal of ancestral lines that will always be too far away
    if pn in [0, 1]:
        # these have no descendants
        return 0
    if iteration_of_next_child is None:
        iteration_checking_at = get_iteration_born_from_point_number(pn)
    else:
        iteration_checking_at = iteration_of_next_child - 1
    adjacency_at_iter = get_adjacency_from_point_number(pn, iteration_checking_at)
    if pn < 12:
        adjacency_at_iter = unify_five_and_six(adjacency_at_iter, pn)
    l, dl, d, _, _, _ = adjacency_at_iter
    xyz = get_xyz_from_point_number(pn)
    distance_to_l = get_distance_point_number_to_xyz_great_circle(l, xyz)
    distance_to_dl = get_distance_point_number_to_xyz_great_circle(dl, xyz)
    distance_to_d = get_distance_point_number_to_xyz_great_circle(d, xyz)
    return radius * max(distance_to_l, distance_to_dl, distance_to_d)


def get_distance_point_number_to_xyz_great_circle(pn, xyz, radius=1):
    xyz2 = get_xyz_from_point_number(pn)
    return UnitSpherePoint.distance_great_circle_xyz_static(xyz, xyz2, radius=radius)


def get_distance_point_codes_great_circle(pc1, pc2, radius=1):
    xyz1 = get_xyz_from_point_code(pc1)
    xyz2 = get_xyz_from_point_code(pc2)
    return UnitSpherePoint.distance_great_circle_xyz_static(xyz1, xyz2, radius=radius)


def get_distance_point_numbers_great_circle(pn1, pn2, radius=1):
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


def plot_directional_parent_graph(iteration):
    g = nx.DiGraph()
    dpar_by_point, children_by_dpar = get_dpar_dicts_up_to_iteration(iteration, first_dchildren_only=True)
    for pc, dpar in dpar_by_point.items():
        if dpar is not None:
            g.add_edge(dpar, pc)
    nx.draw(g, with_labels=True)
    plt.show()


def get_region_around_point_code_by_spreading(pc, max_distance_gc_normalized, resolution_iterations=None):
    # follow adjacency paths at this iteration resolution until you get every point within the radius
    if resolution_iterations is not None:
        pc_iterations = get_iteration_number_from_point_code(pc)
        if pc_iterations > resolution_iterations:
            raise ValueError("center point has more iterations than desired resolution")
        pc += "0" * (resolution_iterations - pc_iterations)
    
    print(f"getting region around {pc} of radius {max_distance_gc_normalized} and resolution of {resolution_iterations} iterations")
    res = {pc}
    new_points = [pc]
    known_neighbors = {}
    checked_points = set()
    t0 = time.time()
    while len(new_points) > 0:
        pc1 = new_points[0]
        if pc1 in known_neighbors:
            neighbors = known_neighbors[pc1]
        else:
            neighbors = get_unordered_neighbors_from_point_code(pc1)
            known_neighbors[pc1] = neighbors
        # print(f"got neighbors of {pc1}: {neighbors}")
        for neighbor in neighbors:
            if neighbor is None:
                continue
            if neighbor not in res and neighbor not in checked_points:
                d = get_distance_point_codes_great_circle(pc, neighbor, radius=1)
                if d <= max_distance_gc_normalized:
                    # print(f"adding {neighbor} to new points")
                    new_points.append(neighbor)
                    res.add(neighbor)
                else:
                    pass # print(f"not adding neighbor {neighbor} because distance {d} is too far (need <= {max_distance_gc_normalized})")
        new_points = new_points[1:]
        checked_points.add(pc1)
        if time.time() - t0 > 2:
            print(f"creating region around {pc}: {len(res)} points so far")
            t0 = time.time()
    print(f"got {len(res)} points in region: {res}")
    return res


def get_region_around_point_code_by_narrowing(pc, max_distance_gc_normalized, narrowing_iterations=None, resolution_iterations=None):
    if narrowing_iterations is None:
        narrowing_iterations = get_iteration_number_from_point_code(pc)
    if resolution_iterations is None:
        resolution_iterations = get_iteration_number_from_point_code(pc) + 1
    inside, outside, split = narrow_watersheds_by_distance(pc, max_distance_gc_normalized, narrowing_iterations)
    res = []
    for wpc in inside:
        res += get_descendants_of_point_code(wpc, resolution_iterations)
    n = len(split)
    t0 = time.time()
    for i, wpc in enumerate(split):
        candidates = get_descendants_of_point_code(wpc, resolution_iterations)
        m = len(candidates)
        for j, pc1 in enumerate(candidates):
            d = get_distance_point_codes_great_circle(pc, pc1)
            if d <= max_distance_gc_normalized:
                res.append(pc1)
            if time.time() - t0 >= 2:
                print(f"checking points in split watersheds for inclusion in region: {i=}/{n=}, {j=}/{m=}")
                t0 = time.time()
    print(f"got {len(res)} points in region: {res}")
    return res


def get_descendants_of_point_code(pc, max_iterations):
    starting_iteration = get_iteration_number_from_point_code(pc)
    if max_iterations < starting_iteration:
        raise ValueError(f"{pc=} of iteration {starting_iteration} has no descendants at iteration {max_iterations}")
    elif max_iterations == starting_iteration:
        # base case of empty causes problems, I think it should be itself only
        # len 1 also makes sense because the next iterations are len 4 ** k
        return [pc]
    else:
        parents = [pc]
        for i in range(starting_iteration + 1, max_iterations + 1):
            children = []
            for pc1 in parents:
                c1 = get_children_from_point_code(pc1)
                children += c1
            parents = children
        return children


def find_distances_to_watershed(reference_pc, watershed_parent_pc, plot=False):
    # find nearest and farthest points within a given watershed
    # can do some geometry like it's one of the corners or it's along a certain edge (binary search?)
    # distance from a certain reference point
    # use this to determine whether we should even consider that watershed when finding points in region
    rpc = reference_pc
    pc0 = watershed_parent_pc

    if pc0[0] in ["A", "B"]:
        assert all(y == "0" for y in pc0[1:]), f"got reverse-encoded or invalid watershed parent {pc0}"
        d = get_distance_point_codes_great_circle(rpc, pc0)
        dmin = d
        dmax = d
        return dmin, dmax

    reference_peel = bc.get_peel_containing_point_code(pc0)  # for re-encoding reversed edges
    # print(f"{rpc=}, {pc0=}")
    watershed_adj = get_adjacency_from_point_code(pc0)  # keep trailing zeros, maybe we want e.g. J00X
    pc1, pc2, pc3, _, _, _ = watershed_adj
    watershed_corners = [pc0, pc1, pc2, pc3]
    # print("corner pcs:", watershed_corners)
    # ds_to_corners = [get_distance_point_codes_great_circle(rpc, pc) for pc in watershed_corners]
    # print("distances to corners:", ds_to_corners)

    # we know that anything in the interior of the watershed
    # must be bounded by the min and max distances to the edges
    # ! IF the reference point is outside the watershed
    # but maybe one of those extrema is in the middle of an edge rather than on a corner

    # I think there should be at most one critical point of distance along a given edge
    # so it will have only one local extremum if any? (endpoints don't count as local extrema here)

    edges = [
        {"p0": pc0, "p1": pc1, "c": "r"},  # edge 1
        {"p0": pc0, "p1": pc3, "c": "b"},  # edge 3
        {"p0": pc3, "p1": pc2, "c": "k"},  # edge 1-opposite
        {"p0": pc1, "p1": pc2, "c": "g"},  # edge 3-opposite
    ]

    n_bits = 4
    binary_arrays = get_binary_arrays(n_bits)

    if plot:
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        rll = get_latlon_from_point_code(rpc)

    if point_is_descendant(rpc, watershed_parent_pc):
        dmin = 0  # the watershed contains the reference point
    else:
        dmin = None
    dmax = None

    for edge_dict in edges:
        # print(f"{edge_dict=}")
        p0 = edge_dict["p0"]
        p1 = edge_dict["p1"]

        switch_edge_due_to_pole = p0[0] in ["A", "B"]
        if switch_edge_due_to_pole:
            assert p1[0] not in ["A", "B"], f"can't have both {p0=} and {p1=} be in the pseudo-watershed of a pole"
            # switch them so we can define the direction along the edge
            p0, p1 = p1, p0

        direction = get_direction_between_point_codes(p0, p1)
        if direction in ["-1", "-2", "-3"]:
            p0, p1 = p1, p0
            direction = get_direction_between_point_codes(p0, p1)
        assert direction in ["1", "2", "3"], f"{p0} -> {p1} gave {direction=}"

        bit_symbols = ["0", direction]
        color = edge_dict["c"]
        pcs = []
        for arr in binary_arrays:
            pc = p0 + "".join(bit_symbols[bit] for bit in arr)
            if bc.point_code_is_in_reversed_polarity_encoding(pc):
                pc = bc.correct_reversed_edge_polarity(pc, reference_peel)
            pcs.append(pc)
        pcs.append(p1)
        ds = [get_distance_point_codes_great_circle(rpc, pc) for pc in pcs]
        dmin = min(dmin, min(ds)) if dmin is not None else min(ds)
        dmax = max(dmax, max(ds)) if dmax is not None else max(ds)
        if plot:
            lls = [get_latlon_from_point_code(pc) for pc in pcs]
            # print("----")
            # for pc, d in zip(pcs, ds):
            #     print(pc, d)
            ax1.plot(ds, c=color)
            ax2.scatter([rll[1]], [rll[0]], marker="x", c="purple")
            ax2.scatter([ll[1] for ll in lls], [ll[0] for ll in lls], c=color)
    
    if plot:
        text = lambda name, pc, ax=ax2: (lambda ll: ax.text(ll[1]+5, ll[0], f"{pc} ({name})"))(get_latlon_from_point_code(pc))
        for name,pc in [("ref",rpc), ("0",pc0), ("1",pc1), ("2",pc2), ("3",pc3)]:
            text(name, pc)
        ax2.set_aspect("equal")
        print(f"{dmin=}, {dmax=}")
        plt.show()
    
    return dmin, dmax


def narrow_watersheds_by_distance(pc, d, max_iterations):
    # go through the watersheds starting from the initial points
    # treat the poles each as a watershed consisting only of a single point

    print(f"narrowing watersheds within {d=} of {pc}")
    watersheds_to_check = list("ABCDEFGHIJKL")
    inside = []
    outside = []
    split = []
    while len(watersheds_to_check) > 0:
        wpc = watersheds_to_check[0]
        dmin, dmax = find_distances_to_watershed(pc, wpc)
        print(f"{wpc=} has {dmin=}, {dmax=}")
        if dmin > d:
            outside.append(wpc)
        elif dmax <= d:
            inside.append(wpc)
        else:
            # it's split, some of it is within the distance and some is not
            children = get_children_from_point_code(wpc)
            if get_iteration_number_from_point_code(wpc) < max_iterations:
                for child in children:
                    watersheds_to_check.append(child)
            else:
                # if the children would have too many iterations
                # then don't split watershed any more
                split += children
        watersheds_to_check = watersheds_to_check[1:]
    return inside, outside, split


def point_is_descendant(pc, ancestor_pc):
    return pc.startswith(ancestor_pc)


def get_direction_between_point_codes(p0, p1):
    # only works when one is in the other's adjacency
    adj = get_adjacency_from_point_code(p0)
    try:
        i = adj.index(p1)
        return ["1", "2", "3", "-1", "-2", "-3"][i]
    except ValueError:
        raise Exception(f"{p1=} not found as neighbor of {p0=}, which has neighbors {adj}")


def get_binary_arrays(n_bits):
    # e.g. if n_bits is 3, return [(0, 0, 0), (0, 0, 1), (0, 1 ,0), etc.]
    if n_bits < 1:
        raise ValueError(n_bits)
    elif n_bits == 1:
        return [[0], [1]]
    else:
        prev = get_binary_arrays(n_bits - 1)
        res = []
        for i in [0, 1]:
            res += [[i] + arr for arr in prev]
        return res


def plot_adjacency_on_map(pc):
    # for debugging
    adj = get_adjacency_from_point_code(pc)
    lat0, lon0 = get_latlon_from_point_code(pc)
    lls = [get_latlon_from_point_code(pc1) for pc1 in adj]
    colors = ["yellow", "red", "blue", "purple", "green", "orange"]
    lats = [ll[0] for ll in lls]
    lons = [ll[1] for ll in lls]
    plt.scatter([lon0], [lat0], c="k")
    plt.scatter(lons, lats, c=colors)
    plt.gca().set_aspect("equal")
    plt.show()


def plot_parents_on_map(pc):
    # for debugging
    par, dpar = get_parents_from_point_code(pc)
    lat, lon = get_latlon_from_point_code(pc)
    plat, plon = get_latlon_from_point_code(par)
    dplat, dplon = get_latlon_from_point_code(dpar)
    plt.scatter([lon], [lat], c="k")
    plt.text(lon, lat, f"{pc}(0)")
    plt.scatter([plon], [plat], c="r")
    plt.text(plon, plat, f"{par}(p)")
    plt.scatter([dplon], [dplat], c="g")
    plt.text(dplon, dplat, f"{dpar}(dp)")
    plt.gca().set_aspect("equal")
    plt.show()


def plot_watershed_inclusion_in_region(inside, outside, split):
    inside_color = "r"
    outside_color = "k"
    split_color = "g"
    inside_pcs = []
    outside_pcs = []
    split_pcs = []
    for wpc in inside:
        # only get the direct children (so 1 iteration later than this point code)
        children = get_children_from_point_code(wpc)
        inside_pcs += children
    for wpc in outside:
        children = get_children_from_point_code(wpc)
        outside_pcs += children
    for wpc in split:
        children = get_children_from_point_code(wpc)
        split_pcs += children
    for pc_list, color in zip([inside_pcs, outside_pcs, split_pcs], [inside_color, outside_color, split_color]):
        lls = [get_latlon_from_point_code(pc) for pc in pc_list]
        lats = [ll[0] for ll in lls]
        lons = [ll[1] for ll in lls]
        plt.scatter(lons, lats, c=color)
    plt.gca().set_aspect("equal")
    plt.show()


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


def test_adjacency_recursive_vs_arithmetic():
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
        adj_test = get_adjacency_from_point_code(pc, iteration, use_old_method=False)
        adj_known = get_adjacency_from_point_code(pc, iteration, use_old_method=True)
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


def test_speed_adjacency_recursive_vs_arithmetic(iterations):
    n = get_n_points_from_iterations(iterations)
    assert n > 2
    print_every_n_seconds = 2

    funcs = {
        "adjacency_recursive": lambda pc, iteration: get_adjacency_from_point_code(pc, iteration, use_old_method=True),
        "adjacency_arithmetic": lambda pc, iteration: get_adjacency_from_point_code(pc, iteration, use_old_method=False),
    }

    for func_name, func in funcs.items():
        t0 = time.time()
        last_print_time = t0
        i = 2
        while i < n:
            if time.time() - last_print_time > print_every_n_seconds:
                print(f"i = {i}/{n}")
                last_print_time = time.time()
            pc = get_point_code_from_point_number(i).ljust(iterations, "0")
            iteration = get_iteration_number_from_point_code(pc)
            adj = func(pc, iteration)
            i += 1
        dt = time.time() - t0
        print(f"{func_name} took {dt} seconds")


def test_directional_parent_position_in_adjacency():
    # it should always be the same as the point's own child index
    pn = 12
    n = get_n_points_from_iterations(5)
    d = {}
    while pn < n:
        child_index = get_child_index_from_point_number(pn)
        dpar = get_directional_parent_from_point_number(pn)
        adj = get_adjacency_from_point_number(pn)
        dpar_direction = adj.index(dpar) if dpar in adj else None
        tup = (child_index, dpar_direction)
        if tup not in d:
            d[tup] = 0
        d[tup] += 1
        if pn % 100 == 0:
            print(f"{pn=}/{n}")
        pn += 1
    print("(child_index, dpar_direction) : number of points like this")
    for k,v in sorted(d.items()):
        print(k, v)


def test_spreading_vs_narrowing():
    while True:
        min_pc_iterations = random.randint(0, 3)
        max_pc_iterations = min_pc_iterations + random.randint(0, 2)
        expected_pc_iterations = (min_pc_iterations + max_pc_iterations) / 2
        pc = get_random_point_code(min_pc_iterations, expected_pc_iterations, max_pc_iterations)
        pc_iterations = get_iteration_number_from_point_code(pc)
        resolution_iterations = pc_iterations + random.randint(2, 4)
        narrowing_iterations = min(resolution_iterations - 1, random.randint(0, 3))
        max_distance_gc_normalized = abs(np.random.normal(0, 0.5))
        # inside, outside, split = narrow_watersheds_by_distance(pc, max_distance_gc_normalized, max_iterations=4)
        # plot_watershed_inclusion_in_region(inside, outside, split)
        t0 = time.time()
        region1 = get_region_around_point_code_by_spreading(pc, max_distance_gc_normalized, resolution_iterations)
        t1 = time.time() - t0
        t0 = time.time()
        region2 = get_region_around_point_code_by_narrowing(pc, max_distance_gc_normalized, narrowing_iterations, resolution_iterations)
        t2 = time.time() - t0
        r1not2 = sorted(set(region1) - set(region2))
        r2not1 = sorted(set(region2) - set(region1))
        print("points from spreading but not narrowing:", r1not2)
        print("points from narrowing but not spreading:", r2not1)
        print(f"spreading took {t1} seconds")
        print(f"narrowing took {t2} seconds")
        scatter_icosa_points_by_code(region1, show=False, marker="o", facecolors="none", edgecolors="b")
        scatter_icosa_points_by_code(region2, show=False, marker="x")
        plt.show()
        print()


STARTING_POINTS = get_starting_points_immutable()  # since this is called way too many times otherwise, just initialize it as a global constant that can be accessed by further functions, e.g. base case for recursive adjacency algorithm
STARTING_POINTS_ORDERED, STARTING_POINTS_ADJACENCY = STARTING_POINTS


if __name__ == "__main__":
    # test_spreading_vs_narrowing()

    # TODO: find latlon using trig, 
    # like the point code tells you how far along a certain edge the point is located
    # e.g. K022020022 is some proportion along the great-circle curve between K and I

    # TODO constrain how much closer/farther a point in the middle of an edge could be
    # also constrain which edges we even need to subdivide in watershed distance measuring
    # this can help us do watershed narrowing more efficiently
    # don't need to check some number of points on all four edges
