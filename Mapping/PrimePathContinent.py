import numpy as np
import matplotlib.pyplot as plt
import math

from PlottingUtil import plot_z_values

import sys
sys.path.insert(0, "/home/kuhron/programming/NumberTheory/")
from PrimeTurtle import get_path_of_numbers


def make_elevation_plot_point_density(xs, ys, grid_unit_size, radius_to_look_for_points, radius_to_elevate, overall_d_el):
    # raise the elevation around each point so areas that have more points are highlands
    min_x = min(xs)
    min_y = min(ys)

    # normalize to first quadrant
    xs = [x - min_x for x in xs]
    ys = [y - min_y for y in ys]

    assert min(xs) == 0 and min(ys) == 0
    x_range = max(xs)
    y_range = max(ys)

    grid_n_xs = math.ceil(x_range / grid_unit_size)
    grid_n_ys = math.ceil(y_range / grid_unit_size)
    grid = np.zeros((grid_n_xs, grid_n_ys))
    grid += overall_d_el

    x_refs = [grid_x * grid_unit_size for grid_x in range(grid_n_xs)]
    y_refs = [grid_y * grid_unit_size for grid_y in range(grid_n_ys)]

    x_slices, y_slices = make_x_and_y_radius_slices(x_refs, y_refs, xs, ys, radius_to_look_for_points)

    for grid_x in range(grid_n_xs):
        print(f"{grid_x = }/{grid_n_xs}")
        x_ref = grid_x * grid_unit_size
        for grid_y in range(grid_n_ys):
            y_ref = grid_y * grid_unit_size
            p_xs, p_ys = get_points_within_radius(x_ref, y_ref, x_slices, y_slices, radius_to_look_for_points)
            n = len(p_xs)
            d_el = n
            grid[grid_x, grid_y] += d_el

    els = grid.T
    im = plot_z_values(els, x_min=-1, x_max=grid_n_xs, y_min=-1, y_max=grid_n_ys)
    plt.show()


def make_x_and_y_radius_slices(x_refs, y_refs, xs, ys, r):
    # make sublists of the points at xs, ys
    # such that for any x_ref in x_refs, we can just go get the list of points within r of that x coordinate
    # and similar for y_ref, and then take their intersection
    print("making radius slices")
    x_slices = {}
    y_slices = {}
    ps = list(zip(xs, ys))
    for i, x_ref in enumerate(x_refs):
        if i % 10 == 0:
            print(f"{i = }/{len(x_refs)}")
        x_slices[x_ref] = set((x,y) for x,y in ps if abs(x - x_ref) <= r)
    for y_ref in y_refs:
        y_slices[y_ref] = set((x,y) for x,y in ps if abs(y - y_ref) <= r)
    print("done making radius slices")
    return x_slices, y_slices


def get_points_within_radius(x_ref, y_ref, x_slices, y_slices, r):
    # first narrow to a square
    candidates_by_x = x_slices[x_ref]
    candidates_by_y = y_slices[y_ref]
    candidates = candidates_by_x & candidates_by_y

    new_xs = []
    new_ys = []
    circle = False
    if circle:
        # now do circle within that
        for x,y in candidates:
            d2 = (x - x_ref)**2 + (y - y_ref)**2
            if d2 <= r**2:
                new_xs.append(x)
                new_ys.append(y)
    else:
        for x,y in candidates:
            new_xs.append(x)
            new_ys.append(y)
    return new_xs, new_ys


if __name__ == "__main__":
    n_min = 1
    n_max = 10**7
    angle_deg = 80
    xs, ys = get_path_of_numbers(n_min=n_min, n_max=n_max, angle_deg=angle_deg)

    grid_unit_size = 50  # how many x/y units apart to put the lattice points for the map (check the axes on the image of the turtle path you specified (via n_min, n_max, and angle_deg) so you know how big it is)
    radius_to_look_for_points = grid_unit_size * 2
    radius_to_elevate = grid_unit_size * 1
    overall_d_el = 0
    make_elevation_plot_point_density(xs, ys, grid_unit_size, radius_to_look_for_points, radius_to_elevate, overall_d_el)

