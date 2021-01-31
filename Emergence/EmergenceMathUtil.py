import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import time
import math



def get_neighbor_sum(arr):
    convolution = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1],
    ])
    return convolve(arr, convolution, mode="same")
    # the convolution array is placed at each point and multiplied by the value at that point, and added to the result
    # if arr is all 0 or 1, then this will give the array of neighbor counts (good for Conway's game of life)
    # mode="same" means the output shape will be the same as the input one, because otherwise the convolution array will spill over the edges


def get_power_of_r_sums(arr, power):
    # e.g. power=-2 will give the 1/r^2 sum at each point of all the other points
    # the convolution needs to be able to sit at any corner and cover the whole array
    dx, dy = arr.shape  # assumes 2D
    # so if arr is say 2 by 3, need 3 by 5 (2n-1)
    # and then the center coordinates in this array will be (1,2) = (n-1)/2
    big_dx = 2*dx-1
    big_dy = 2*dy-1
    center = [(big_dx-1)//2, (big_dy-1)//2]
    convolution_shape = (big_dx, big_dy)
    # getting some functions from here: https://stackoverflow.com/questions/40820955/numpy-average-distance-from-array-center
    grid_x, grid_y = np.mgrid[0:big_dx, 0:big_dy]
    # shift the coordinates to being centered on the big array center
    grid_x = grid_x - center[0]
    grid_y = grid_y - center[1]
    r = np.sqrt(grid_x**2 + grid_y**2)
    assert r[0,0] == r[-1,0] == r[0,-1] == r[-1,-1], "four corners should have same value in radius array"
    assert r[center[0], center[1]] == 0, "center of radius array should be zero"
    r_to_power = r**power
    if power <= 0:
        # replace the center with 0 because it'll be inf
        r_to_power[center[0], center[1]] = 0
    return convolve(arr, r_to_power, mode="same")  # mode=same means output is same shape as input, with convolution not spilling over the edges but just centered on each point

