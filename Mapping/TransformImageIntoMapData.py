# take an image, e.g. a map of Myst labeled as land or sea
# and the corner coordinates of the image's location on the globe
# get a lattice (IcosahedralGeodesicLattice) of desired granularity
# put the data from the image on the corresponding nearby lattice points
# e.g. make a data file (csv from pandas df) with all the lattice's point indices
# and corresponding data from the image (if any)
# should use a dictionary of image point color to data value, and should know the variable name being used

# desired output e.g.
# index,land_or_sea
# 0,na
# 1,na
# 2,land
# 3,sea
# 4,na
# 5,land
# ...

# and then later can use these dfs directly without having to load the images anymore
# but do keep the image files so you can make new things to import, easier to do it by drawing on the image in paint so you know where you're putting stuff e.g. volcanoes


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from Lattice import Lattice
from LatitudeLongitudeLattice import LatitudeLongitudeLattice
from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
import MapCoordinateMath as mcm


def get_rgba_color_array_from_image_array(arr):
    # collapse the last dimension into tuples; maybe np has a better way to do this but I can't find it
    color_arr = np.empty(shape=arr.shape[:-1], dtype=tuple)
    assert len(color_arr.shape) == 2, color_arr.shape
    for xi in range(color_arr.shape[0]):
        for yi in range(color_arr.shape[1]):
            color_arr[xi,yi] = tuple(arr[xi,yi,:])
    return color_arr


if __name__ == "__main__":
    test_input_fp = "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_elevation_Mako.png"
    latlon00 = [10,10]
    latlon01 = [10,30]
    latlon10 = [-10,10]
    latlon11 = [-10,30]

    lattice = IcosahedralGeodesicLattice(iterations=6)
    im = Image.open(test_input_fp)
    width, height = im.size

    image_lattice = LatitudeLongitudeLattice(
        height, width,  # rows, columns
        latlon00, latlon01, latlon10, latlon11
    )  # we are not actually going to add data to this lattice, but we will use it to get point coordinates more easily

    arr = np.array(im)
    print(arr)

    # just translate the colors into strings

    color_to_str = {
        (0, 255, 255, 255): "cyan",
        (0, 0, 255, 255): "blue",
        (255, 255, 255, 255): "white",
    }

    assert arr.shape[-1] == 4, arr.shape  # RGBA dimension

    color_arr = get_rgba_color_array_from_image_array(arr)
    str_arr = np.empty(shape=arr.shape[:-1], dtype=str)
    for color, val in color_to_str.items():
        str_arr[color_arr == color] = val
    print(str_arr)


