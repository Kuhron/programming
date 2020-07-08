import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_land_and_sea_colormap():
    # see PrettyPlot.py
    linspace_cmap_forward = np.linspace(0, 1, 128)
    linspace_cmap_backward = np.linspace(1, 0, 128)
    blue_to_black = mcolors.LinearSegmentedColormap.from_list('BlBk', [
        mcolors.CSS4_COLORS["blue"], 
        mcolors.CSS4_COLORS["black"],
    ])
    land_colormap = mcolors.LinearSegmentedColormap.from_list('land', [
        mcolors.CSS4_COLORS["darkgreen"],
        mcolors.CSS4_COLORS["limegreen"],
        mcolors.CSS4_COLORS["gold"],
        mcolors.CSS4_COLORS["darkorange"],
        mcolors.CSS4_COLORS["red"],
        mcolors.CSS4_COLORS["saddlebrown"],
        mcolors.CSS4_COLORS["gray"],
        mcolors.CSS4_COLORS["white"],
        # mcolors.CSS4_COLORS[""],
    ])
    # colors_land = plt.cm.YlOrBr(linspace_cmap_backward)  # example of how to call existing colormap object
    colors_land = land_colormap(linspace_cmap_forward)
    colors_sea = blue_to_black(linspace_cmap_backward)
    colors = np.vstack((colors_sea, colors_land))
    colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return colormap


def get_contour_levels(min_elevation, max_elevation):
    # print("getting contour levels from elevation limits min = {}, max = {}".format(min_elevation, max_elevation))
    n_sea_contours = 20
    n_land_contours = 100

    # the elevations on both sides of zero have to be equidistant or the map's land/sea cutoff will be in the wrong place (at elevation != 0), so need to make average of min and max value = 0.
    # don't care much about very deep sea, so just take max land value as max abs, and make min value its negative
    if max_elevation <= 0:
        print("Warning: max elevation non-positive: {}; moving it to zero".format(max_elevation))
        max_elevation = 0
    min_elevation = -max_elevation
    # print("new contour elevation limits min = {}, max = {}".format(min_elevation, max_elevation))

    epsilon_elevation = 0.1

    if min_elevation < 0:
        sea_contour_levels = np.linspace(min_elevation, -1*epsilon_elevation, n_sea_contours)
    else:
        sea_contour_levels = [-1*epsilon_elevation]
    if max_elevation > 0:
        land_contour_levels = np.linspace(epsilon_elevation, max_elevation, n_land_contours)
    else:
        land_contour_levels = [epsilon_elevation]
    contour_levels = list(sea_contour_levels) + [0] + list(land_contour_levels)
    # print("sea contour levels: {}".format(sea_contour_levels))
    # print("land contour levels: {}".format(land_contour_levels))
    # for i, level in enumerate(contour_levels):
    #     print("contour level {} = {}".format(i, level))
    return contour_levels



