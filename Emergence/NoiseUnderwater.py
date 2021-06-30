import math
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

import NoiseMath as nm

import sys
sys.path.insert(0,'..')  # cause I can't be bothered to make packages for all these separate things
from Distortion01 import DistortionFunction01, DistortionFunctionSeries01
from CubicXYFunction import CubicXYFunction
from PltContentOnly import imshow_content_only



def add_line(arr, xy_mesh, add_at_nodes=False, warp=False, warp_stdev=0.5):
    if warp:
        xy_mesh = warp_xy_mesh(xy_mesh, warp_stdev)
        # print("new xy_mesh:")
        # print(xy_mesh)
        # input("press enter")

    # style = random.choice(["line", "cubic_curve"])
    style = "cubic_curve"
    if style == "line":
        func_vals = nm.get_distances_from_line(xy_mesh)
    elif style == "cubic_curve":
        # center it on (0.5, 0.5), center of the 01 box
        f = CubicXYFunction.random(stdev=1)
        x,y = xy_mesh
        func_vals = f(x-0.5,y-0.5)
        # center the z value on 0
        func_vals -= np.mean(func_vals)
    else:
        raise ValueError(f"unknown style {style}")

    bump_func = nm.get_bump_func()  # raise around areas where the function is zero
    bumped_arr = bump_func(func_vals)

    if add_at_nodes:
        # lines will add together where they overlap
        arr += bumped_arr
    else:
        arr = np.maximum(arr, bumped_arr)
    return arr


def warp_xy_mesh(xy_mesh, stdev):
    fx = DistortionFunctionSeries01.random(stdev)
    fy = DistortionFunctionSeries01.random(stdev)
    x, y = xy_mesh
    x = fx(x)
    y = fy(y)
    return x,y


def get_interesting_array():
    arr = np.zeros((1000, 1000))
    xy_mesh = nm.get_xy_mesh(arr)  # just pass this around since it won't ever change

    for i in range(6):
        arr = add_line(arr, xy_mesh, 
            add_at_nodes=False, 
            warp=False, warp_stdev=0.9,
        )

    return arr


def generate_images(n_images):
    for im_i in range(n_images):
        a = get_interesting_array()
        now_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_fp = f"NoiseImages/Underwater/{now_str}-{im_i}.png"
        imshow_content_only(a, save_fp=out_fp)


if __name__ == "__main__":
    generate_images(10)
