import math
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# plot curves by having black line where lhs==rhs and continuous coloring elsewhere

def f1(x, y, *args):
    r = args[0]

    lhs = x
    rhs = r * np.arccos(1-y/r) - np.sqrt(y*(2*r-y))

    print("f1; r = {r}".format(r=r).replace("; ", "\n   ")+"\n")
    return lhs - rhs

def f2(x, y, *args):
    # elliptic curve
    a = args[0]
    b = args[1]

    lhs = y**2
    rhs = x**3 + a*x + b

    print("f2; a = {a}; b = {b}".format(a=a, b=b).replace("; ", "\n   ")+"\n")
    return lhs - rhs

def f3(x, y, *args):
    a = args[0]
    b = args[1]

    lhs = y**2
    rhs = x**5 + a*abs(x)**0.5 + b

    print("f3; a = {a}; b = {b}".format(a=a, b=b).replace("; ", "\n   ")+"\n")
    return lhs - rhs

def f4(x, y, *args):
    n = len(args[0])
    a = args[0][:n]
    b = args[1][:n]
    c = args[2][:n]
    d = args[3][:n]
    e = args[4][:n]

    lhs = 0
    rhs = 0

    for i in range(len(a)):
        # the result is the sum of this many sine waves in 2d
        rhs += a[i]*np.sin(b[i]*x) + c[i]*np.sin(d[i]*y) + e[i]

    print("f4 ({n} waves); a = {a}; b = {b}; c = {c}; d = {d}; e = {e}".format(a=a, b=b, c=c, d=d, e=e, n=n).replace("; ", "\n   ")+"\n")
    return lhs - rhs

def f5(x, y, *args):
    n = len(args[0])
    a = args[0][:n]
    b = args[1][:n]
    c = args[2][:n]
    d = args[3][:n]
    e = args[4][:n]

    lhs = 0
    rhs = 1

    for i in range(len(a)):
        # the result is the product of this many sine waves in 2d
        rhs *= a[i]*np.sin(b[i]*x) + c[i]*np.sin(d[i]*y) + e[i]

    print("f5 ({n} waves); a = {a}; b = {b}; c = {c}; d = {d}; e = {e}".format(a=a, b=b, c=c, d=d, e=e, n=n).replace("; ", "\n   ")+"\n")
    result = lhs - rhs
    return signed_log(result)

def f6(x, y, *args):
    n = len(args[0])
    a = args[0][:n]
    b = args[1][:n]
    c = args[2][:n]
    d = args[3][:n]
    e = args[4][:n]

    lhs = 0
    rhs = 1

    for i in range(len(a)):
        # the result is the difference between the sum and the product of this many sine waves in 2d
        lhs += c[i]*np.sin(d[i]*x) + a[i]*np.sin(b[i]*y) + e[i]
        rhs *= a[i]*np.sin(b[i]*x) + c[i]*np.sin(d[i]*y) + e[i]

    print("f6 ({n} waves); a = {a}; b = {b}; c = {c}; d = {d}; e = {e}".format(a=a, b=b, c=c, d=d, e=e, n=n).replace("; ", "\n   ")+"\n")
    result = lhs - rhs
    return signed_log(result)

def signed_log(x):
    return np.log(1+abs(x)) * np.sign(x)

def plot_deviation(f, *args):
    xlim = (-3, 3)
    ylim = (-3, 3)
    extent = list(xlim) + list(ylim)
    n_points = 1000
    xs = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0])/n_points)
    ys = np.arange(ylim[0], ylim[1], (ylim[1] - ylim[0])/n_points)

    X, Y = np.meshgrid(xs, ys[::-1])
    Z = f(X, Y, *args)

    zero_contour_width_ratio = 1
    zlim = (np.min(Z), np.max(Z))
    n_levels = 20
    zs = np.arange(zlim[0], zlim[1], (zlim[1] - zlim[0])/n_levels)
    widths = [0.5]*len(zs) + [zero_contour_width_ratio * 0.5]
    zs = np.concatenate([zs, [0]])
    z_width_dict = {z: w for z, w in zip(zs, widths)}  # removes duplicate heights while making sure zero gets the thick width
    z_w_tuples = sorted((z, w) for z, w in z_width_dict.items())
    zs, widths = [tuple(x) for x in zip(*z_w_tuples)]

    contours_misplaced = False
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    def get_land_and_sea_colormap():
        # stacking 2 colormaps, from https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
        # sample the colormaps that you want to use. Use 128 from each so we get 256 colors in total
        colors_land = plt.cm.autumn(np.linspace(1, 0, 128))  # autumn: yellow for low-elevation land, to red for high-elevation
        colors_sea = plt.cm.winter(np.linspace(0, 1, 128))  # winter: green for shallow sea, to blue for deep
        # combine them and build a new colormap
        colors = np.vstack((colors_sea, colors_land))  # goes from low to high, so put sea first
        colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        return colormap

    # colormap = "Spectral"
    colormap = get_land_and_sea_colormap()
    max_color_value = 3
    min_color_value = -1 * max_color_value  # keep 0 in the middle of the colormap

    if contours_misplaced:
        im = plt.imshow(Z, cmap=colormap, extent=extent, vmin=min_color_value, vmax=max_color_value)
        cs = plt.contour(Z, extent=extent, colors="black", levels=zs, linewidths=widths)
    else:
        im = plt.imshow(Z, cmap=colormap, extent=extent, origin="lower", vmin=min_color_value, vmax=max_color_value)
        cs = plt.contour(Z, extent=extent, colors="black", levels=zs, linewidths=widths, origin="lower")
    
    # plt.clabel(cs, inline=False, fmt='%1.1f', fontsize=np.nan, levels=[0])
    plt.colorbar(im)
    plt.show()


if __name__ == "__main__":
    # plot_deviation(f2, random.uniform(-1, 1), random.uniform(-1, 1))
    # plot_deviation(f3, random.uniform(-1, 1), random.uniform(-1, 1))
    n_f4_waves = np.random.randint(4, 8)  # right-exclusive; for f4, too many doesn't do a lot except increase magnitude of oscillations
    n_f5_waves = np.random.randint(4, 8)  # for f5, too many makes way too much of the map close to zero, with random huge peaks and troughs
    n_f6_waves = np.random.randint(7, 16)  # for f6, too many makes the f5 have almost no effect except for random huge peaks and troughs
    plot_deviation(f4,
        np.random.uniform(-1, 1, n_f4_waves),
        np.random.uniform(-4, 4, 100),
        np.random.uniform(-1, 1, 100),
        np.random.uniform(-4, 4, 100),
        np.random.uniform(-1, 1, 100)
    )
    plot_deviation(f5,
        np.random.uniform(-2, 2, n_f5_waves),
        np.random.normal(0, 2, 100),
        np.random.uniform(-2, 2, 100),
        np.random.normal(0, 2, 100),
        np.random.uniform(-1, 1, 100)
    )
    plot_deviation(f6,
        np.random.uniform(-2, 2, n_f6_waves),
        np.random.normal(0, 2, 100),
        np.random.uniform(-2, 2, 100),
        np.random.normal(0, 2, 100),
        np.random.uniform(-1, 1, 100)
    )


