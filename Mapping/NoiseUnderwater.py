import math
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.insert(0,'..')  # cause I can't be bothered to make packages for all these separate things
from Distortion01 import DistortionFunction01, DistortionFunctionSeries01


def add_straight_line(arr, xy_mesh, add_at_nodes=False, warp=False, warp_stdev=0.5):
    x0,y0,x1,y1 = np.random.uniform(0, 1, (4,))
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])

    # get distance of each point from this line, apply a bump-like function to make a ridge
    if warp:
        xy_mesh = warp_xy_mesh(xy_mesh, warp_stdev)
        # print("new xy_mesh:")
        # print(xy_mesh)
        # input("press enter")
    distances = distance_to_line_two_point_form(p0, p1, xy_mesh)

    bump_width = np.random.lognormal(np.log(0.05), np.log(2))
    bump_func = lambda distance: 1/bump_width * np.maximum(0, bump_width - distance)  # height of 1
    bumped_arr = bump_func(distances)

    if add_at_nodes:
        # lines will add together where they overlap
        arr += bumped_arr
    else:
        arr = np.maximum(arr, bumped_arr)
    return arr


def get_xy_mesh(arr):
    # assumes 01 box
    x_size, y_size = arr.shape
    xs = np.linspace(0,1,x_size)
    ys = np.linspace(0,1,y_size)
    return np.meshgrid(xs, ys)


def distance_to_line_two_point_form(line_point_0, line_point_1, query_point):
    p0 = line_point_0
    p1 = line_point_1
    pa = query_point
    x0,y0 = p0
    x1,y1 = p1
    xa,ya = pa
    dx = x1-x0
    dy = y1-y0
    d01 = math.sqrt(dx**2 + dy**2)

    # two transformations: translate so p0 is at origin (T1), then rotate so p1 is on y-axis (T2)
    # composition of these transformations makes it so pa's x-coordinate is its distance from the now-vertical line (which is on y-axis)
    # T1 = (x0,y0) -> (0, 0) = (-x0, -y0); p1 -> p1' = (x1', y1') = (x1-x0, y1-y0); pa -> pa' = (xa-x0, ya-y0)
    # T2 is rotation matrix with tan theta = x1'/y1'; Rot = (1,0) -> (cos theta, sin theta), (0,1) -> (-sin theta, cos theta) := [[c,-s],[s,c]] (row-major)
    # cos arctan x = 1/sqrt(1+x^2); sin arctan x = x/sqrt(1+x^2) (drew a triangle)
    # so c = dy/d01; s = dx/d01; verified that Rot*p1' = (0, d01)
    # Rot*pa' gives x of (xa'y1' - x1'ya')/d01
    xap = xa-x0
    yap = ya-y0
    x1p = dx
    y1p = dy
    xa_after_transformations = (xap*y1p - x1p*yap)/d01
    return abs(xa_after_transformations)


def warp_xy_mesh(xy_mesh, stdev):
    fx = DistortionFunctionSeries01.random(stdev)
    fy = DistortionFunctionSeries01.random(stdev)
    x, y = xy_mesh
    x = fx(x)
    y = fy(y)
    return x,y


def get_interesting_array():
    arr = np.zeros((1000, 1000))
    xy_mesh = get_xy_mesh(arr)  # just pass this around since it won't ever change

    for i in range(30):
        arr = add_straight_line(arr, xy_mesh, 
            add_at_nodes=True, 
            warp=True, warp_stdev=0.6,
        )

    return arr


if __name__ == "__main__":
    a = get_interesting_array()
    plt.imshow(a)
    plt.colorbar()
    plt.show()
