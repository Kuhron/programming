# try to decrease the amount of memory used up by large lattices
# keep info about Icosa coordinates in files, look things up in tables, rather than keeping it all in RAM


import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt


def get_latlon_from_point_number(point_number):
    raise NotImplementedError


def get_xyz_from_point_number(point_number):
    raise NotImplementedError


def get_iterations_from_number_of_points(n):
    try:
        # n_points(n_iters) = 10 * (4**n_iters) + 2
        return {12: 0, 42: 1, 162: 2, 642: 3, 2562: 4, 10242: 5}[n]
    except KeyError:
        iterations = get_exact_iterations_from_points(n)
        assert iterations % 1 == 0, "number of points {} gave non-int number of iterations; make sure it is 2+10*(4**n)".format(n)
        return iterations


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

    plt.subplot(1,3,1)
    plt.scatter(point_numbers, xs)

    plt.subplot(1,3,2)
    plt.scatter(point_numbers, ys)

    plt.subplot(1,3,3)
    plt.scatter(point_numbers, zs)

    plt.show()


def plot_latlons(n_iterations):
    d = get_position_memo_dict(n_iterations)
    n_points = get_exact_points_from_iterations(n_iterations)
    point_numbers = range(12, n_points)
    lats = [d[pi]["latlondeg"][0] for pi in point_numbers]
    lons = [d[pi]["latlondeg"][1] for pi in point_numbers]
    
    plt.subplot(1,2,1)
    plt.scatter(point_numbers, lats)

    plt.subplot(1,2,2)
    plt.scatter(point_numbers, lons)

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


if __name__ == "__main__":
    n_iterations = 7
    # plot_coordinate_patterns(n_iterations)

    point_numbers = [40, 171, 5009]
    print(get_specific_adjacencies(point_numbers, n_iterations))

