import os
import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import MapCoordinateMath as mcm
from LatitudeLongitudeLattice import LatitudeLongitudeLattice
from TransformImageIntoMapData import get_image_fp_to_latlon, shrink_resolution


def get_lattice_and_array_from_image(image_fp, latlon00, latlon01, latlon10, latlon11):
    im = Image.open(image_fp)
    im = shrink_resolution(im)
    width, height = im.size
    image_lattice = LatitudeLongitudeLattice(
        height, width,
        latlon00, latlon01, latlon10, latlon11,
    )
    arr = np.array(im)
    return image_lattice, arr


def get_latlon_to_color_dict(image_fp, latlon00, latlon01, latlon10, latlon11):
    lattice, image_arr = get_lattice_and_array_from_image(image_fp, latlon00, latlon01, latlon10, latlon11)
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


def get_xyrgba_array(image_fp, latlon00, latlon01, latlon10, latlon11):
    print("- getting xyrgba array for {}".format(image_fp))
    latlon_to_color_dict = get_latlon_to_color_dict(image_fp, latlon00, latlon01, latlon10, latlon11)
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


def plot_images_on_globe(image_fp_to_latlon, save_fp=None):
    full_xyrgba_array = []
    for image_fp, latlons in image_fp_to_latlon.items():
        latlon00, latlon01, latlon10, latlon11 = latlons
        xyrgba_array = get_xyrgba_array(image_fp, latlon00, latlon01, latlon10, latlon11)
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
    plt.show()


if __name__ == "__main__":
    image_location_data_fp = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/ImageToLocationDict.csv"
    image_fp_to_latlon = get_image_fp_to_latlon(image_location_data_fp)
    save_fp = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/ContinentsPlacedOutput.png"
    if os.path.exists(save_fp):
        input("Warning, file exists and will be overwritten by plot: {}\npress enter to continue".format(save_fp))
    plot_images_on_globe(image_fp_to_latlon, save_fp=save_fp)
