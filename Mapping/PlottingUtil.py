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
    n_sea_contours = 20
    n_land_contours = 100
    if min_elevation < 0:
        sea_contour_levels = np.linspace(min_elevation, 0, n_sea_contours)
    else:
        sea_contour_levels = [0]
    if max_elevation > 0:
        land_contour_levels = np.linspace(0, max_elevation, n_land_contours)
    else:
        land_contour_levels = [0]
    assert sea_contour_levels[-1] == land_contour_levels[0] == 0
    contour_levels = list(sea_contour_levels[:-1]) + list(land_contour_levels)
    return contour_levels



