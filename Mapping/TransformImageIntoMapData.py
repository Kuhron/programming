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
    color_arr = np.empty(shape=arr.shape[:-1], dtype=object)  # object is better dtype so np not trying to iterate
    assert len(color_arr.shape) == 2, color_arr.shape
    for xi in range(color_arr.shape[0]):
        for yi in range(color_arr.shape[1]):
            color_arr[xi,yi] = tuple(arr[xi,yi,:])
    return color_arr


def get_condition_string_array_from_image_array(arr, color_to_str):
    color_arr = get_rgba_color_array_from_image_array(arr)
    str_arr = np.empty(shape=color_arr.shape, dtype=object)
    # needs to be dtype object, not str or tuple, so np not trying to iterate (it will just put the first char of the str if you do that)
    # something like str_arr[color_arr == color] would be nice, but it complains about elementwise comparison and doesn't give the correct result (they all come up false)
    for color, val in color_to_str.items():
        for i in range(color_arr.shape[0]):
            for j in range(color_arr.shape[1]):
                color_in_img = color_arr[i,j]
                s = color_to_str.get(color_in_img)
                if s is not None:
                    str_arr[i,j] = s
    return str_arr


if __name__ == "__main__":
    test_input_fp = "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_elevation_Mako.png"
    # test_input_fp = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/MientaDigitization_ThinnedBorders_Final.png"
    latlon00 = [30,-30]
    latlon01 = [30,30]
    latlon10 = [-30,-30]
    latlon11 = [-30,30]

    map_lattice = IcosahedralGeodesicLattice(iterations=6)
    im = Image.open(test_input_fp)
    width, height = im.size

    image_lattice = LatitudeLongitudeLattice(
        height, width,  # rows, columns
        latlon00, latlon01, latlon10, latlon11
    )  # we are not actually going to add data to this lattice, but we will use it to get point coordinates more easily

    arr = np.array(im)
    assert arr.shape[-1] == 4, arr.shape  # RGBA dimension

    # just translate the colors into strings
    color_to_str = {
        (0, 255, 255, 255): "cyan",
        (0, 38, 255, 255): "blue",
        (255, 255, 255, 255): "white",
        (255, 0, 0, 255): "red",
    }

    str_arr = get_condition_string_array_from_image_array(arr, color_to_str)
    print(np.unique(str_arr))

    # now get map lattice points which are inside the image lattice
    image_lattice_points_in_order = image_lattice.points
    str_arr_in_order = str_arr.reshape((str_arr.size,))
    str_at_image_lattice_point = dict(zip(image_lattice_points_in_order, str_arr_in_order))
    point_values_to_assign = {p_i: None for p_i in range(len(map_lattice.points))}
    for p in map_lattice.points:
        in_image = image_lattice.contains_point_latlon(p)
        if in_image:
            closest_image_point = image_lattice.closest_point_to(p)
            value_to_assign = str_at_image_lattice_point[closest_image_point]
            p_i = map_lattice.get_index_of_usp(p)
            point_values_to_assign[p_i] = value_to_assign
            print("point {} now has value {}".format(p_i, value_to_assign))

    # for each of those, associate it with the image lattice point which is closest to it
    # then assign the image lattice point's color/str to the map lattice point
    # then write these map lattice data strs to database file by point index

    df = map_lattice.create_dataframe()
    condition_labels = []
    with open("TestTransformImageIntoMapDataResult.txt", "w") as f:
        for p_i, val in sorted(point_values_to_assign.items()):
            f.write("{},{}\n".format(p_i, val))
            condition_labels.append(val)
    df["condition_label"] = condition_labels

    category_labels = [None] + sorted(color_to_str.values())
    map_lattice.plot_data(df, "condition_label", category_labels=category_labels, equirectangular=True)
    plt.show()
