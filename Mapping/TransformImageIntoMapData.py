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
import datetime
import time
import random
import pandas as pd

from Lattice import Lattice
from LatitudeLongitudeLattice import LatitudeLongitudeLattice
from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
import IcosahedronMath
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
    r_size, c_size, rgba_len = arr.shape
    assert rgba_len == 4
    str_arr = [["" for c in range(c_size)] for r in range(r_size)]
    # I'm gonna try using a python list instead of np stuff because dtype=object makes it too huge in memory
    # if using np array, needs to be dtype object, not str or tuple, so np not trying to iterate (it will just put the first char of the str if you do that)
    # something like str_arr[color_arr == color] would be nice, but it complains about elementwise comparison and doesn't give the correct result (they all come up false)
    for r in range(r_size):
        for c in range(c_size):
            color = color_arr[r,c]
            # assert type(color_in_img) is tuple  # this must be true since it's a dict key, don't bother checking each time
            try:
                s = color_to_str[color]
            except KeyError:
                print(f"Warning: Color {color} was found in the image, but it is not in the color_to_str dict.")
                s = input("Please type a value to be used for this color (default empty string): ").strip()
                color_to_str[color] = s
            str_arr[r][c] = s
    return str_arr


def get_lattice_from_image(image_fp, latlon00, latlon01, latlon10, latlon11, color_to_str, key_str, with_coords=False):
    im = Image.open(test_input_fp)
    width, height = im.size

    image_lattice = LatitudeLongitudeLattice(
        height, width,  # rows, columns
        latlon00, latlon01, latlon10, latlon11
    )

    arr = np.array(im)
    assert arr.shape[-1] == 4, arr.shape  # RGBA dimension
    str_arr = get_condition_string_array_from_image_array(arr, color_to_str)

    df = image_lattice.create_dataframe(with_coords=with_coords)

    df_index = df.index
    point_indices = image_lattice.get_point_indices()
    assert len(df_index) == len(point_indices)
    assert list(df_index) == list(point_indices)
    df[key_str] = ["" for p_i in point_indices]
    original_df_size = df.size

    print("strings found in image:", np.unique(str_arr))
    print("creating image lattice")
    for x in range(image_lattice.x_size):
        print(f"row {x}/{image_lattice.x_size} ({100*x/image_lattice.x_size :.2f}%)")
        assert df.size == original_df_size, "df growing out of control"  # catching assignment bug that tries to make new column/multi-index on every assignment
        for y in range(image_lattice.y_size):
            # color = arr[x,y]
            point_number = image_lattice.lattice_position_to_point_number[(x,y)]
            # print("point number is {}".format(point_number))
            str_val = str_arr[x][y]
            df[key_str][point_number] = str_val
            # print("str_val is {}".format(str_val))
            # print("df size is now {}".format(df.size))

    print("done creating image lattice")
    return image_lattice, df


def find_icosa_points_near_image_lattice_points(image_lattice, icosa_point_tolerance, planet_radius):
    STARTING_POINTS = IcosahedronMath.STARTING_POINTS

    print("finding icosa points near image lattice points")
    d = {}
    for x in range(image_lattice.x_size):
        print(f"row {x}/{image_lattice.x_size} ({100*x/image_lattice.x_size :.2f}%)")
        for y in range(image_lattice.y_size):
            point_number = image_lattice.lattice_position_to_point_number[(x,y)]
            point = image_lattice.points[point_number]
            latlon = point.latlondeg()
            nearest_icosa_point, distance_normalized, distance_km = IcosahedronMath.get_nearest_icosa_point_to_latlon(latlon, icosa_point_tolerance, planet_radius, STARTING_POINTS)
            # print(f"\nimage ({x}, {y}); point #{point_number}; point {point}; icosa point within {icosa_point_tolerance_km} km = {nearest_icosa_point} at distance of {distance_km} km away")
            d[point_number] = nearest_icosa_point
    print("done finding icosa points near image lattice points")
    return d


def write_image_conditions_lattice_agnostic(image_fp, output_fp, latlon00, latlon01, latlon10, latlon11, color_to_str, key_str, icosa_distance_normalized):
    print("writing image conditions df to {}".format(output_fp))
    image_lattice, df = get_lattice_from_image(test_input_fp, latlon00, latlon01, latlon10, latlon11, color_to_str, key_str, with_coords=False)
    print("getting approx icosa points for image pixels")
    approx_icosa_point_numbers = []
    for pi, p in enumerate(image_lattice.points):
        if pi % 1000 == 0 and pi != 0:
            print(f"point {pi}/{image_lattice.n_points} ({100*pi/image_lattice.n_points :.2f}%)")
        approx_p, d_norm, d_units = IcosahedronMath.get_nearest_icosa_point_to_latlon(p.latlondeg(), icosa_distance_normalized, planet_radius=1, STARTING_POINTS=IcosahedronMath.STARTING_POINTS)
        approx_p_i = approx_p.point_number
        approx_icosa_point_numbers.append(approx_p_i)
    print("done getting approx icosa points for image pixels, adding them to the df")
    unique_icosa_point_numbers = np.unique(approx_icosa_point_numbers)
    all_unique = len(unique_icosa_point_numbers) == len(approx_icosa_point_numbers)
    if not all_unique:
        print(f"Warning: not all unique icosa points. Got {len(unique_icosa_point_numbers)} unique values for {len(approx_icosa_point_numbers)} points")
    df["approx_icosa_point_number"] = approx_icosa_point_numbers
    print("calling df.to_csv()")
    df.to_csv(output_fp, chunksize=1000)
    print("done writing image conditions df to {}".format(output_fp))


if __name__ == "__main__":
    test_input_fp = "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_elevation_Ilausa.png"
    # test_input_fp = "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_elevation_Circle.png"
    # test_input_fp = "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_elevation_Mako.png"
    # test_input_fp = "/home/wesley/programming/Mapping/Projects/CadaTest/ImageImporting/EGII_CadaTest_volcanism_Mako.png"
    # test_input_fp = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/MientaDigitization_ThinnedBorders_Final.png"
    latlon00 = [50.01,-120.4]
    latlon01 = [50.02,-110.1]
    latlon10 = [0.1,-120.4]
    latlon11 = [0.1,-110.1]

    # just translate the colors into strings
    color_to_str = {
        (0, 255, 255, 255): "sea",
        (0, 38, 255, 255): "coast",
        (255, 255, 255, 255): "land",
        (255, 0, 0, 255): "land",
    }

    key_str = "elevation"

    icosa_point_tolerance_km = 1
    planet_radius_km = IcosahedronMath.CADA_II_RADIUS_KM
    icosa_distance_normalized = icosa_point_tolerance_km / planet_radius_km

    latlon_to_corner_coord_str = lambda latlon: "{},{}".format(*latlon)
    corner_coords_str = ";".join(latlon_to_corner_coord_str(latlon) for latlon in [latlon00, latlon01, latlon10, latlon11])
    resolution_str = "cada{}km".format(icosa_point_tolerance_km)
    test_output_fp_info_str = f"_{key_str}_{corner_coords_str}_{resolution_str}.csv"
    test_output_fp = test_input_fp.replace(".png", test_output_fp_info_str)
    assert test_output_fp != test_input_fp, "input_fp didn't contain .png"
    write_image_conditions_lattice_agnostic(test_input_fp, test_output_fp, latlon00, latlon01, latlon10, latlon11, color_to_str, key_str, icosa_distance_normalized)
    input("got here")

    # image_lattice, df = get_lattice_from_image(test_input_fp, latlon00, latlon01, latlon10, latlon11, color_to_str, key_str)
    # icosa_points = find_icosa_points_near_image_lattice_points(image_lattice, icosa_point_tolerance_km, planet_radius_km)

    icosa_iterations_of_precision = IcosahedronMath.get_iterations_needed_for_edge_length(icosa_point_tolerance_km, planet_radius_km)
    n_icosa_points = IcosahedronMath.get_points_from_iterations(icosa_iterations_of_precision)

    # now get map lattice points which are inside the image lattice
    image_lattice_points_in_order = image_lattice.points
    point_values_to_assign = {}
    t0 = time.time()
    for p_i in range(n_icosa_points):
        if p_i % 1000 == 0 and p_i != 0:
            progress_proportion = p_i / n_icosa_points
            elapsed = time.time() - t0
            rate = progress_proportion / elapsed
            remaining_proportion = 1 - progress_proportion
            eta = remaining_proportion / rate
            eta_str = str(datetime.timedelta(seconds=eta))
            print("icosa point {} of {} ({}%), ETA {}".format(p_i, n_icosa_points, 100*progress_proportion, eta_str))
        usp = IcosahedronMath.get_usp_from_point_number(p_i)
        in_image = image_lattice.contains_point_latlon(usp)
        if in_image:
            closest_image_point_index, closest_image_point = image_lattice.closest_point_to(usp)
            value_to_assign = df[key_str][closest_image_point_index]
            point_values_to_assign[p_i] = value_to_assign
            # print("point {} now has value {}".format(p_i, value_to_assign))

    # for each of those, associate it with the image lattice point which is closest to it
    # then assign the image lattice point's color/str to the map lattice point
    # then write these map lattice data strs to database file by point index

    # map_df = map_lattice.create_dataframe()
    # condition_labels = []
    # with open("TestTransformImageIntoMapDataResult.txt", "w") as f:
    #     for p_i, val in sorted(point_values_to_assign.items()):
    #         f.write("{},{}\n".format(p_i, val))
    #         condition_labels.append(val)
    # map_df["condition_label"] = condition_labels

    # category_labels = [None] + sorted(color_to_str.values())
    # map_lattice.plot_data(df, "condition_label", category_labels=category_labels, equirectangular=True)
    # plt.show()
