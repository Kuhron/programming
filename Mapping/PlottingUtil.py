import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
import scipy


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
    colormap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

    # https://stackoverflow.com/questions/11386054/python-matplotlib-change-default-color-for-values-exceeding-colorbar-range
    colormap.set_over("white")
    colormap.set_under("black")
    # but note that set_over and set_under won't work for contourf (which simply draws *nothing* in regions where the values are out of range), workaround is ax.set_facecolor(<out-of-range color>)
    return colormap


def get_volcanism_colormap():
    linspace_cmap_forward = np.linspace(0, 1, 128)
    linspace_cmap_backward = np.linspace(1, 0, 128)
    forward_cmap = plt.get_cmap("hot")
    forward_colors = forward_cmap(linspace_cmap_forward)
    # backward_colors = [invert_hue(c) for c in forward_colors][::-1]  # this goes black, light_blue, dark_blue, white; the blues are reversed from what I want
    backward_cmap = mcolors.LinearSegmentedColormap.from_list('x', [
        mcolors.CSS4_COLORS["black"],
        # mcolors.CSS4_COLORS["blue"],  # too dark
        "#0070FF",
        mcolors.CSS4_COLORS["cyan"],
        mcolors.CSS4_COLORS["white"],
    ])
    backward_colors = backward_cmap(linspace_cmap_backward)

    colors = np.vstack((backward_colors, forward_colors))
    colormap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
    return colormap


def invert_hue(rgba):
    # approximation I'm making up, not sure if it actually correctly inverts hue
    # flip each of r, g, and b over WITHIN the interval (min_magnitude, max_magnitude)
    # so black maps to black, white maps to white, and e.g. (0.7, 0.5, 0.4) maps to (0.4, 0.6, 0.7)
    r, g, b, a = rgba
    min_mag = min(r, g, b)
    max_mag = max(r, g, b)
    r2 = min_mag + (max_mag - r)
    g2 = min_mag + (max_mag - g)
    b2 = min_mag + (max_mag - b)
    return np.array([r2, g2, b2, a])


def get_contour_levels(min_value, max_value, prefer_positive=False, n_sea_contours=20, n_land_contours=100):
    if prefer_positive:
        # print("getting contour levels from elevation limits min = {}, max = {}".format(min_elevation, max_elevation))
        min_elevation = min_value
        max_elevation = max_value
    
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

    else:
        # symmetrical between positive and negative
        n_contours_each_sign = n_land_contours
        max_abs = max(abs(min_value), abs(max_value))
        # in case it's all zero, still want levels to be increasing so it doesn't throw error
        max_abs = max(max_abs, 1)
        contour_levels = np.linspace(-1*max_abs, max_abs, 2*n_contours_each_sign + 1)  # +1 so there is zero in the middle

    return contour_levels


def plot_interpolated_data(data_coords, values, lat_range, lon_range, n_lats, n_lons, with_axis=False):
    min_lat, max_lat = lat_range if lat_range is not None else (-90, 90)
    min_lon, max_lon = lon_range if lon_range is not None else (-180, 180)
    interpolation_lats = np.linspace(min_lat, max_lat, n_lats)
    interpolation_lons = np.linspace(min_lon, max_lon, n_lons)
    interpolation_grid_latlon = np.array(list(itertools.product(interpolation_lats, interpolation_lons)))
    print(f"interpolation lats has shape {interpolation_lats.shape}")
    print(f"interpolation lons has shape {interpolation_lons.shape}")
    print(f"interpolation grid has shape {interpolation_grid_latlon.shape}")

    # interpolate
    interpolated = scipy.interpolate.griddata(data_coords, values, interpolation_grid_latlon, method="linear")
    print(f"interpolated has shape {interpolated.shape}")
    len_interp, = interpolated.shape
    len_lats, = interpolation_lats.shape
    len_lons, = interpolation_lons.shape
    assert len_lats * len_lons == len_interp

    # make grid that imshow will like
    # x is longitude, y is latitude
    n_rows = len_lats
    n_cols = len_lons
    Z = np.empty((n_rows, n_cols), dtype=float)
    for id_lat, lat in enumerate(interpolation_lats):
        row_number = id_lat
        for id_lon, lon in enumerate(interpolation_lons):
            col_number = id_lon
            value_index = row_number * (n_cols) + col_number
            assert (interpolation_grid_latlon[value_index] == (lat, lon)).all(), f"transposition error at row,col ({row_number}, {col_number}), which is latlon ({lat}, {lon})"
            value = interpolated[value_index]
            Z[row_number, col_number] = value

    if with_axis:
        fig, ax = plt.subplots()
        fig.set_size_inches(8,4)
        fig.add_axes(ax)
    else:
        # plot without any axes or frame
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.set_size_inches(8,4)
        ax.set_axis_off()
        fig.add_axes(ax)

    min_value = Z[np.isfinite(Z)].min()
    max_value = Z[np.isfinite(Z)].max()
    contourf_levels = get_contour_levels(min_value, max_value, prefer_positive=True, n_sea_contours=20, n_land_contours=100)
    cmap = get_land_and_sea_colormap()
    im = ax.contourf(Z, origin="lower", extent=[min_lon, max_lon, min_lat, max_lat], levels=contourf_levels, cmap=cmap)  # imshow extent is left,right,bottom,top
    plt.xlim(min_lon, max_lon)
    plt.ylim(min_lat, max_lat)
    if with_axis:
        plt.colorbar(im)


