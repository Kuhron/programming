import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import MapCoordinateMath as mcm
from LatitudeLongitudeLattice import LatitudeLongitudeLattice


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


def shrink_resolution(im):
    # so giant images don't take up huge amounts of memory just for the purposes of plotting where they go on the globe (high resolution not necessary for this)
    print("Notice: shrinking resolution of image. If you do not want this, remove calls to shrink_resolution(im)")
    max_len = 100
    r,c = im.size
    new_r = min(max_len, r)
    new_c = min(max_len, c)
    r_factor = new_r/r
    c_factor = new_c/c
    # choose the SMALLER factor and shrink the whole image that amount
    factor = min(r_factor, c_factor)
    new_r = int(r*factor)
    new_c = int(r*factor)
    im = im.resize((new_r, new_c), Image.ANTIALIAS)
    return im


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
    legron_latlons = [(41,-130), (33,-50), (-38,-130), (-38.1,-50)]
    mienta_latlons = [(30,62), (30.1,118), (-30.1,62), (-30,118)]
    oz_latlons = [(-35.1,-160), (-30,-38), (-35.1,145), (-35,20)]
    it_latlons = [(67,-125), (67,125), (45, -20), (45,40)]
    si_latlons = [(-11.2, 154.3), (-11.2, 155.4), (-12.4, 154.1), (-12.4, 155.7)]  # Sertorisun Islands can take up approx 1 sq degree, giving area similar to West Virginia
    image_fp_to_latlon = {
        # "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_elevation_Ilausa.png" : [(10,-10),(10,10.1),(-10.1,-9.9),(-10,10)],
        # "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_elevation_Circle.png": [(65,-100),(70,-90),(55,-95),(60,-85)],
        # "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_elevation_Mako.png": [(-40,-40),(-40,-20),(-60,-49),(-59,-29)],
        # "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_volcanism_Mako.png": [(50,50),(50,55),(45,50),(45,55.1)],
        "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/LegronCombinedDigitization_ThinnedBorders_Final.png": legron_latlons,
        "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/MientaDigitization_ThinnedBorders_Final.png": mienta_latlons,
        "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/OligraZitomoDigitization_ThinnedBorders_Final.png": oz_latlons,
        "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/ImisTolinDigitization_ThinnedBorders_Final.png": it_latlons,
        "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/SertorisunIslandsDigitization_ThinnedBorders_Final.png": si_latlons,
    }
    save_fp = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/output.png"
    if os.path.exists(save_fp):
        input("Warning, file exists and will be overwritten by plot: {}\npress enter to continue".format(save_fp))
    plot_images_on_globe(image_fp_to_latlon, save_fp=save_fp)
