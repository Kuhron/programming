import os
import csv
from PIL import Image
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy

import MapCoordinateMath as mcm
from LatitudeLongitudeLattice import LatitudeLongitudeLattice
from TransformImageIntoMapData import get_image_fp_to_latlon, shrink_resolution


def get_lattice_and_array_from_image(image_fp, latlon00, latlon01, latlon10, latlon11, shrink=True):
    im = Image.open(image_fp)
    if shrink:
        im = shrink_resolution(im)
    width, height = im.size
    image_lattice = LatitudeLongitudeLattice(
        height, width,
        latlon00, latlon01, latlon10, latlon11,
    )
    arr = np.array(im)
    return image_lattice, arr


def get_latlon_to_color_dict(image_fp, latlon00, latlon01, latlon10, latlon11, shrink=True):
    lattice, image_arr = get_lattice_and_array_from_image(image_fp, latlon00, latlon01, latlon10, latlon11, shrink=shrink)
    r_size, c_size, rgba_len = image_arr.shape
    assert r_size == lattice.r_size
    assert c_size == lattice.c_size
    assert rgba_len == 4
    
    d = {}
    print("- constructing latlon to color dict")
    latlon_array = lattice.get_latlondegs_array()
    for r in range(r_size):
        if r % 10 == 0:
            print("row {}/{}".format(r, r_size))
        for c in range(c_size):
            # p_i = lattice.get_point_number_from_lattice_position(r, c)
            # p = lattice.get_point_from_point_number(p_i)
            # latlon = p.latlondeg()
            latlon = latlon_array[:,r,c]
            color = image_arr[r,c]
            assert latlon.shape == (2,), latlon.shape
            latlon = tuple(latlon)
            d[latlon] = color
    print("- done constructing latlon to color dict")
    return d


def get_xyrgba_array(image_fp, latlon00, latlon01, latlon10, latlon11, shrink=True):
    print("- getting xyrgba array for {}".format(image_fp))
    latlon_to_color_dict = get_latlon_to_color_dict(image_fp, latlon00, latlon01, latlon10, latlon11, shrink=shrink)
    lst = []
    for latlon, color_tup in latlon_to_color_dict.items():
        lat, lon = latlon
        r, g, b, a = color_tup
        tup = (lat, lon, r, g, b, a)
        lst.append(tup)
    print("- done getting xyrgba array for {}".format(image_fp))
    return lst


def add_opaque_background(image_fp):
    # https://stackoverflow.com/questions/50898034/how-replace-transparent-with-a-color-in-pillow
    im = Image.open(image_fp)
    color = "BLACK"
    image = Image.new("RGB", im.size, color)
    image.paste(im, (0, 0), im) 
    image.save(image_fp)


def plot_images_on_globe_scatter(image_fp_to_latlon, save_fp=None, show=True, shrink=True):
    full_xyrgba_array = []
    for image_fp, latlons in image_fp_to_latlon.items():
        latlon00, latlon01, latlon10, latlon11 = latlons
        xyrgba_array = get_xyrgba_array(image_fp, latlon00, latlon01, latlon10, latlon11, shrink=shrink)
        full_xyrgba_array.extend(xyrgba_array)
    arr = np.array(full_xyrgba_array)
    n_points, six = arr.shape
    assert six == 6, arr.shape
    lats = arr[:,0]
    lons = arr[:,1]
    colors = arr[:, 2:]
    colors /= 255

    # plot without any axes or frame
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])

    fig.set_size_inches(8,4)
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.scatter(lons, lats, c=colors)
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.rcParams['figure.facecolor'] = "black"
    plt.rcParams['savefig.facecolor'] = "black"
    if save_fp is not None:
        plt.savefig(save_fp, facecolor="black")
        add_opaque_background(save_fp)  # because matplotlib facecolor is being a huge pain and never works
    if show:
        plt.show()


def plot_images_on_globe_imshow(image_fp_to_latlon, save_fp=None, show=True, shrink=True):
    full_xyrgba_array = []
    for image_fp, latlons in image_fp_to_latlon.items():
        latlon00, latlon01, latlon10, latlon11 = latlons
        xyrgba_array = get_xyrgba_array(image_fp, latlon00, latlon01, latlon10, latlon11, shrink=shrink)
        full_xyrgba_array.extend(xyrgba_array)
    arr = np.array(full_xyrgba_array)
    n_points, six = arr.shape
    assert six == 6, arr.shape
    # lats = arr[:,0]
    # lons = arr[:,1]
    data_coords = arr[:,:2]  # lat then lon
    colors = arr[:, 2:]
    colors /= 255

    interpolation_lons = np.arange(-180, 180, 1)
    interpolation_lats = np.arange(-90, 90, 1)
    interpolation_grid_latlon = np.array(list(itertools.product(interpolation_lats, interpolation_lons)))
    print(f"interpolation lats has shape {interpolation_lats.shape}")
    print(f"interpolation lons has shape {interpolation_lons.shape}")
    print(f"interpolation grid has shape {interpolation_grid_latlon.shape}")
    values = np.array([np.linalg.norm(c) for c in colors])  # just pick something for now so it gets float values to interpolate

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

    # plot without any axes or frame
    # fig = plt.figure(frameon=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig, ax = plt.subplots()

    fig.set_size_inches(8,4)
    # ax.set_axis_off()
    fig.add_axes(ax)

    # ax.scatter(lons, lats, c=colors)
    ax.imshow(Z, extent=[-180, 180, 90, -90])  # need y axis backwards since imshow reads rows from top down
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.rcParams['figure.facecolor'] = "black"
    plt.rcParams['savefig.facecolor'] = "black"
    if save_fp is not None:
        plt.savefig(save_fp, facecolor="black")
        add_opaque_background(save_fp)  # because matplotlib facecolor is being a huge pain and never works
    if show:
        plt.show()


if __name__ == "__main__":
    image_location_data_fp = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/ImageToLocationDict.csv"
    image_fp_to_latlon = get_image_fp_to_latlon(image_location_data_fp)
    # save_fp = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/ContinentsPlacedOutput.png"
    save_fp = None
    if save_fp is not None and os.path.exists(save_fp):
        input("Warning, file exists and will be overwritten by plot: {}\npress enter to continue".format(save_fp))

    # plot_images_on_globe_scatter(image_fp_to_latlon, save_fp=save_fp, show=True, shrink=True)
    plot_images_on_globe_imshow(image_fp_to_latlon, save_fp=save_fp, show=True, shrink=True)
