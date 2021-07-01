from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
import IcosahedronMath as im
import NoiseMath as nm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import random

from ImageMetadata import get_image_metadata_dict, get_world_metadata_dict
import PlottingUtil as pu


def get_image_pixel_to_icosa_point_number_from_calculation(image_name, pixels=None):
    # if specify pixels, it will only do those
    print(f"getting image pixel to icosa point number dict for {image_name}")
    metadata = get_image_metadata_dict()[image_name]
    image_fp = metadata["image_fp"]
    icosa_distance_tolerance_normalized = get_icosa_distance_tolerance_normalized(image_name)

    image_lattice = get_lattice_from_image(image_name)
    print(f"image lattice has {image_lattice.n_points} points")

    rc_to_usp = image_lattice.get_points_by_lattice_position(pixels)
    rcs = sorted(rc_to_usp.keys())
    usps = [rc_to_usp[rc] for rc in rcs]

    point_numbers = get_approx_icosa_point_numbers_from_usps(usps, icosa_distance_tolerance_normalized)
    rc_to_point_number = {rc: p_i for rc, p_i in zip(rcs, point_numbers)}  # assumes the orders match, which they should if you get usps in the same order as rcs they correspond to
    print(f"done getting image pixel to icosa point number dict for {image_name}")
    return rc_to_point_number


def get_image_pixel_to_icosa_point_number_from_memo(image_name):
    print(f"reading image pixel to icosa point number correspondence for image_name {image_name}")
    pixel_to_icosa_fp = get_image_metadata_dict()[image_name]["pixel_to_icosa_fp"]
    if not os.path.exists(pixel_to_icosa_fp):
        raise FileNotFoundError(f"pixel to icosa point number file not found: {pixel_to_icosa_fp}")
    with open(pixel_to_icosa_fp) as f:
        lines = f.readlines()
    strs = [l.strip().split(",") for l in lines]
    ints = [[int(x) for x in s] for s in strs]
    
    # convert to dict with (r,c) keys
    r_len = len(ints)
    c_len = len(ints[0])
    n_pixels = r_len * c_len
    assert all(len(x) == c_len for x in ints), "inconsistent column number in the rows of icosa point number array"

    d = {}
    for r in range(r_len):
        for c in range(c_len):
            assert (r,c) not in d
            d[(r,c)] = ints[r][c]
    assert len(d) == n_pixels
    print(f"done reading image pixel to icosa point number correspondence for image_name {image_name}")
    return d


def get_rc_size(rcs):
    max_r = -1
    max_c = -1
    assert all(r >= 0 and c >= 0 for r,c in rcs), "row/column number may not have negative value"
    for r,c in rcs:
        max_r = max(r, max_r)
        max_c = max(c, max_c)
    r_size = max_r + 1
    c_size = max_c + 1
    return r_size, c_size


def get_condition_array_categorical(image_name, map_variable):
    strs = get_condition_array_shorthand(image_name, map_variable)
    # assume all shorthands are non-negative ints, give -1 to the absent condition
    ints = []
    for str_row in strs:
        int_row = []
        for s in str_row:
            if s == "":
                n = -1
            else:
                n = int(s)
                assert n >= 0
            int_row.append(n)
        ints.append(int_row)

    return ints


def get_condition_array_shorthand(image_name, map_variable):
    metadata = get_image_metadata_dict()[image_name]
    condition_array_dir = metadata["condition_array_dir"]

    filename = f"{image_name}_{map_variable}_condition_shorthand.txt"
    fp = os.path.join(condition_array_dir, filename)

    with open(fp) as f:
        lines = f.readlines()
    strs = [l.strip().split(",") for l in lines]
    return strs


def get_condition_shorthand_dict(image_name, map_variable):
    metadata = get_image_metadata_dict()
    condition_array_dir = metadata[image_name]["condition_array_dir"]
    shorthand_filename = f"ImageKey_{map_variable}_condition.csv"
    shorthand_fp = os.path.join(condition_array_dir, shorthand_filename)

    d = {}
    dict_key_colname = "shorthand"
    with open(shorthand_fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["colors_rgba"] = parse_colors_rgba(row["colors_rgba"])
            row["min"] = float(row["min"]) if row["min"] != "" else None
            row["max"] = float(row["max"]) if row["max"] != "" else None
            assert (row["min"] is None or row["max"] is None) or (row["min"] <= row["max"])
            dict_key = row[dict_key_colname]
            assert dict_key not in d
            d[dict_key] = row
    return d


def get_rgba_to_condition_dict(image_name, map_variable):
    shorthand_dict = get_condition_shorthand_dict(image_name, map_variable)
    d = {}
    for sh, row in shorthand_dict.items():
        colors_rgba = row["colors_rgba"]
        for tup in colors_rgba:
            r,g,b,a = tup  # just to check that they're 4-tuples
            assert tup not in d, f"duplicate color {tup}"
            d[tup] = row
    return d


def parse_colors_rgba(rgbas_str):
    rgba_strs = rgbas_str.split(",")
    res = []
    for rgba_str in rgba_strs:
        assert rgba_str[0] == "#"
        rgba_str = rgba_str[1:]
        assert len(rgba_str) == 4*3
        r = int(rgba_str[0:3])
        g = int(rgba_str[3:6])
        b = int(rgba_str[6:9])
        a = int(rgba_str[9:12])
        rgba = (r,g,b,a)
        res.append(rgba)
    return res


def get_default_values_of_conditions(image_name, map_variable):
    condition_shorthand_array = get_condition_array_shorthand(image_name, map_variable)
    shorthand_dict = get_condition_shorthand_dict(image_name, map_variable)
    shorthand_to_default_value = {sh: get_default_value_from_min_and_max(shorthand_dict[sh]["min"], shorthand_dict[sh]["max"]) for sh in shorthand_dict}
    convert = np.vectorize(lambda x: shorthand_to_default_value[x])  # don't do .get because we want to raise for invalid key
    default_values = convert(condition_shorthand_array)
    return default_values


def get_default_value_from_min_and_max(min_val, max_val):
    # if condition has min and max, use average
    # if condition has only min or max, use that +/- 1 on the allowed side (or could just use the extremum itself, since who knows what units that 1 is in)
    # if condition has neither, use 0 (or NaN? but want to seed from the default for elevation, so use 0)
    if min_val is None and max_val is None:
        return 0.0
    elif min_val is None and max_val is not None:
        return max_val - 1.0
    elif max_val is None and min_val is not None:
        return min_val + 1.0
    else:
        return (min_val + max_val)/2.0
    # return all floats because np.vectorize very annoyingly will only choose a single return type; it assumes the output type of the first element in the input is what is desired for all of the inputs. Not true here if we mix floats and ints!


def create_cada_ii_default_value_dict(map_variable):
    # create dict of icosa point : default elevation value (or other map variable)
    print(f"creating Cada II default value dict for {map_variable}")
    image_metadata = get_image_metadata_dict()
    d = {}
    for image_name in image_metadata.keys():
    # for image_name in ["Sertorisun Islands"]:  # debug
        default_value_arr = get_default_values_of_conditions(image_name, map_variable)
        icosa_point_number_dict = get_image_pixel_to_icosa_point_number_from_memo(image_name)
        r_size, c_size = get_rc_size(icosa_point_number_dict.keys())
        for r in range(r_size):
            for c in range(c_size):
                default_value = default_value_arr[r][c]
                icosa_pn = icosa_point_number_dict[(r,c)]
                assert icosa_pn not in d
                d[icosa_pn] = default_value
    print(f"done creating Cada II default value dict for {map_variable}")
    return d


def plot_default_values_by_image():
    image_metadata = get_image_metadata_dict()
    for image_name in image_metadata.keys():
        default_value_arr = get_default_values_of_conditions(image_name, map_variable)
        plt.imshow(default_value_arr)
        plt.colorbar()
        plt.show()


def plot_icosa_point_number_to_value_dict(pn_to_val):
    print("plotting icosa point number to value dict")
    pns = sorted(pn_to_val.keys())
    pns = random.sample(pns, min(len(pns), 10000))  # debug
    latlons = im.get_latlons_from_point_numbers(pns)
    data_coords = latlons
    values = [pn_to_val[pn] for pn in pns]

    lat_range = [-90, 90]
    lon_range = [-180, 180]
    n_lats = 1000
    n_lons = 2 * n_lats
    pu.plot_interpolated_data(data_coords, values, lat_range, lon_range, n_lats, n_lons, with_axis=False)
    print("plotting icosa point number to value dict")
    plt.show()




if __name__ == "__main__":
    image_metadata = get_image_metadata_dict()
    world_metadata = get_world_metadata_dict()["Cada II"]
    print(world_metadata)
    map_variable = "elevation"

    pn_to_default_value = create_cada_ii_default_value_dict(map_variable)
    plot_icosa_point_number_to_value_dict(pn_to_default_value)

    # old stuff
    # df = pd.read_csv(input_fp, index_col="icosa_point_number")
    # n_rows = len(df.index)
    # iterations = im.get_iterations_from_points(n_rows)
    # lattice = IcosahedralGeodesicLattice(iterations=iterations)
    # lattice_df = lattice.create_dataframe()

    # print(lattice_df)
    # input("a")

    # needed_columns = ["usp", "xyz", "latlondeg"]  # things left out of the written df
    # df[needed_columns] = lattice_df[needed_columns]  # populate with values from the lattice computation

    # lattice.plot_data(df, "elevation", equirectangular=True, size_inches=(48, 24))
    # plt.show()

