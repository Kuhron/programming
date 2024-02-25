# measuring the scale of detail in different parts of an image, and the overall distribution of these scales
# from high idea 2024-02-24

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.interpolate import UnivariateSpline


# try defining the amount of detail in a set of pixels as the magnitude of the std or variance of their colors (in some color space where you can define distances between colors)
# so for a given pixel, can measure how much detail is in neighborhoods around it with various radii
# and somehow from that, get a figure for the scale of detail at that point
# detail density in an area = color std / area , has units of u/p**2, where u is the unit of color distance and p is a pixel side length
# can get the value at a point as limit as radius goes to zero of the detail density
# scale of detail can be something like a characteristic length, the p such that u/p**2 is some value that we pick


def color_distance(c0, c1):
    # idea from https://stackoverflow.com/a/47586402/7376935
    r0, g0, b0, a0 = c0
    r1, g1, b1, a1 = c1
    dr = r1 - r0
    dg = g1 - g0
    db = b1 - b0
    da = a1 - a0
    rgb_dist2 = (dr**2 + dg**2 + db**2)/3
    dist2 = da**2 / 2 + rgb_dist2 * a0 * a1 / (255**2)
    dist = dist2**0.5
    return dist


def get_color_std(a):
    r, c, four = a.shape
    assert four == 4, four
    avg = np.mean(a, axis=(0,1))
    assert avg.shape == (four,), avg.shape
    variance = 0
    for r_i in range(r):
        for c_i in range(c):
            color = a[r_i, c_i]
            diff = color_distance(color, avg)
            variance += diff**2
    n = r*c
    variance /= n-1
    std = variance ** 0.5
    return std


def extrapolate_slopes(radii, values):
    # one extrapolation approach, take each slope into account (from x=1 to x=2, from x=1 to x=3, ... from x=1 to x=20) and do a weighted average of them with some falloff function for the farther-out slopes (exponential, or 1/r**2), use this weighted average slope to backtrack from x=1 to x=0 and return that value
    if len(radii) < 2:
        raise ValueError("need slopes to extrapolate from")
    slopes = []
    factors = []
    r1 = radii[0]  # will usually be one, but if we got zero std at r=1 then we can't use that for log_std so we omit the data point and do the best we can from 2 instead
    v1 = values[0]
    discount_func = lambda d: 1/d**2
    for i in range(1, len(radii)):
        r = radii[i]
        v = values[i]
        slope = (v - v1)/(r - r1)
        factor = discount_func(r - r1)
        slopes.append(slope)
        factors.append(factor)
    sumproduct = lambda xs, ys: sum(x*y for x,y in zip(xs, ys))
    m = sumproduct(slopes, factors) / sum(factors)
    # now return a function that can be called at zero
    extrapolator = lambda r: v1 + m*(r - r1)
    return extrapolator


def extrapolate_spline(radii, values):
    # another extrapolation approach, just use an existing spline library, but could generate garbage (but then again probably so could my method)
    extrapolator = UnivariateSpline(radii, values, k=3)
    return extrapolator


def get_detail_density_at_point(r, c, a, r_size, c_size):
    max_radius = 10
    radii = list(range(1, max_radius + 1))
    stds = []
    densities = []
    for radius in radii:
        r_min = max(0, r-radius)
        r_max = min(r_size-1, r+radius)
        c_min = max(0, c-radius)
        c_max = min(c_size-1, c+radius)
        neighborhood = a[r_min:r_max+1, c_min:c_max+1]  # naive, easy first
        color_std = get_color_std(neighborhood)
        stds.append(color_std)
        density = color_std / radius**2
        densities.append(density)

    # log of std makes more sense since std can't go below zero
    # get rid of places where it's zero
    new_radii = []
    log_stds = []
    for r, std in zip(radii, stds):
        if std == 0:
            continue
        new_radii.append(r)
        log_stds.append(math.log(std))
    radii = new_radii

    # try extrapolating the curve and getting a value at zero
    # alternative approach, fix radius and plot the landscape and distribution when you measure each point with that radius of neighborhood

    # extrapolator = extrapolate_spline(radii, log_stds)
    extrapolator = extrapolate_slopes(radii, log_stds)  # yeah my hacky way is looking a lot better than spline for this data
    val0 = extrapolator(0)

    # plt.plot(radii, log_stds)
    # plt.scatter([0,], [val0,], c="r")
    # plt.title(f"{r,c}")
    # plt.show()

    return val0



if __name__ == "__main__":
    im_fp = "/home/kuhron/Desktop/Thoughts/Drugs/2024-02-24/Screenshot from 2024-02-25 03-36-30.png"
    im = Image.open(im_fp)
    a = np.array(im)

    # NB: Pinta shows (column, row) order when you mouseover pixels
    r_size, c_size, four = a.shape
    assert four == 4

    # plot estimated detail density value over the whole image to see how it varies across the different regions
    arr = np.zeros((r_size, c_size))
    vals = []
    for r in range(r_size):
        print(f"getting detail densities: row {r}/{r_size}")
        for c in range(c_size):
            print(f"col {c}/{c_size}", end="\r")
            d = get_detail_density_at_point(r, c, a, r_size, c_size)
            arr[r,c] = d
            vals.append(d)
        print()
    plt.imshow(arr)
    plt.show()

    # and plot a histogram of the values
    plt.hist(vals, bins=100)
    plt.show()

    # TODO this is VERY inefficient, find better ways to compute this value for each pixel or save on duplicated computations
