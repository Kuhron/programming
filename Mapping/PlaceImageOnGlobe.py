import os
import csv
from PIL import Image
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy

import MapCoordinateMath as mcm
import PlottingUtil as pu
from LatitudeLongitudeLattice import LatitudeLongitudeLattice
from TransformImageIntoMapData import shrink_resolution
from ReadMetadata import get_region_metadata_dict, get_latlon_dict


import sys
sys.path.insert(0,'..')  # cause I can't be bothered to make packages for all these separate things
from PltContentOnly import add_opaque_background



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


def get_latlon_to_color_dict(image_name, shrink=True):
    latlon_dict = get_latlon_dict()
    latlon00, latlon01, latlon10, latlon11 = latlon_dict[image_name]
    image_fp = get_region_metadata_dict()[image_name]["image_fp"]

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


def get_xyrgba_array(image_name, shrink=True):
    print("- getting xyrgba array for {}".format(image_name))
    latlon_to_color_dict = get_latlon_to_color_dict(image_name, shrink=shrink)
    lst = []
    for latlon, color_tup in latlon_to_color_dict.items():
        lat, lon = latlon
        r, g, b, a = color_tup
        tup = (lat, lon, r, g, b, a)
        lst.append(tup)
    print("- done getting xyrgba array for {}".format(image_name))
    return lst


def plot_images_on_globe_scatter(image_names, save_fp=None, show=True, shrink=True):
    full_xyrgba_array = []
    for image_name in image_names:
        xyrgba_array = get_xyrgba_array(image_name, shrink=shrink)
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
    if save_fp is not None:
        plt.savefig(save_fp)
        add_opaque_background(save_fp)  # because matplotlib facecolor is being a huge pain and never works
    if show:
        plt.show()


def plot_images_on_globe_imshow(image_names, save_fp=None, show=True, shrink=True, lat_range=None, lon_range=None):
    full_xyrgba_array = []
    for image_name in image_names:
        xyrgba_array = get_xyrgba_array(image_name, shrink=shrink)
        full_xyrgba_array.extend(xyrgba_array)
    arr = np.array(full_xyrgba_array)
    n_points, six = arr.shape
    assert six == 6, arr.shape
    # lats = arr[:,0]
    # lons = arr[:,1]
    data_coords = arr[:,:2]  # lat then lon
    colors = arr[:, 2:]
    colors /= 255

    values = np.array([np.linalg.norm(c) for c in colors])
    # values = np.array([np.random.normal(np.linalg.norm(c),0.05) for c in colors])  # just pick something for now so it gets float values to interpolate

    n_lats = 1000
    n_lons = 2000
    pu.plot_interpolated_data(data_coords, values, lat_range, lon_range, n_lats, n_lons)
    if save_fp is not None:
        plt.savefig(save_fp, facecolor="black")
        add_opaque_background(save_fp)  # because matplotlib facecolor is being a huge pain and never works
    if show:
        plt.show()


if __name__ == "__main__":
    metadata = get_region_metadata_dict()
    image_names = sorted(metadata.keys())
    # save_fp = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/ContinentsPlacedOutput.png"
    save_fp = None
    if save_fp is not None and os.path.exists(save_fp):
        input("Warning, file exists and will be overwritten by plot: {}\npress enter to continue".format(save_fp))

    plot_images_on_globe_scatter(image_names, save_fp=save_fp, show=True, shrink=True)
    # plot_images_on_globe_imshow(image_names, save_fp=save_fp, show=True, shrink=True, lat_range=[-30,30], lon_range=[-150,-90])
