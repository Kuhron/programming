import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import random
import re
from PIL import Image

import IcosahedronMath as icm
import NoiseMath as nm
from Lattice import Lattice
from LatitudeLongitudeLattice import LatitudeLongitudeLattice
import MapCoordinateMath as mcm
from ReadMetadata import get_region_metadata_dict, get_world_metadata_dict, get_latlon_dict, get_icosa_distance_tolerance_normalized
import PlottingUtil as pu



def get_image_pixel_to_icosa_point_code_from_calculation(region_name, pixels=None):
    # if specify pixels, it will only do those
    print(f"getting image pixel to icosa point code dict for {region_name}")
    image_fp = get_region_template_image_fp(region_name)
    icosa_distance_tolerance_normalized = get_icosa_distance_tolerance_normalized(region_name)

    lattice = get_lattice_for_region(region_name)
    print(f"region's template image lattice has {lattice.n_points} points")

    rc_to_usp = lattice.get_points_by_lattice_position(pixels)
    rcs = sorted(rc_to_usp.keys())
    usps = [rc_to_usp[rc] for rc in rcs]

    point_codes = get_approx_icosa_point_codes_from_usps(usps, icosa_distance_tolerance_normalized)
    rc_to_point_code = {rc: p_i for rc, p_i in zip(rcs, point_codes)}  # assumes the orders match, which they should if you get usps in the same order as rcs they correspond to
    print(f"done getting image pixel to icosa point code dict for {region_name}")
    return rc_to_point_code


def get_pixel_to_icosa_fp(region_name):
    root_dir = get_root_dir_for_world_of_region(region_name)
    dirpath = os.path.join(root_dir, "PixelToIcosaPointFiles")
    fname = f"{region_name}_PixelToIcosaPointCode.txt"
    fp = os.path.join(dirpath, fname)
    return fp


def get_image_pixel_to_icosa_point_code_from_memo(region_name):
    # print(f"reading image pixel to icosa point code correspondence for {region_name=}")
    pixel_to_icosa_fp = get_pixel_to_icosa_fp(region_name)
    # print(f"reading memo at {pixel_to_icosa_fp}")
    if not os.path.exists(pixel_to_icosa_fp):
        raise FileNotFoundError(f"pixel to icosa point code file not found: {pixel_to_icosa_fp}")
    with open(pixel_to_icosa_fp) as f:
        lines = f.readlines()
    strs = [l.strip().split(",") for l in lines]
    # print(f"done reading image pixel to icosa point code correspondence for {region_name=}")
    return strs


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


def get_control_condition_dir(world_name):
    root_dir = get_root_dir_for_world(world_name)
    dirpath = os.path.join(root_dir, "ControlConditions")
    return dirpath


def get_condition_image_fp(region_name, map_variable):
    world_name = get_world_name_from_region_name(region_name)
    dirpath = get_control_condition_dir(world_name)
    fname = f"{region_name}_{map_variable}_condition.png"
    fp = os.path.join(dirpath, fname)
    return fp


def get_region_and_world_metadata(region_name):
    region_metadata = get_region_metadata_dict()[region_name]
    world_name = region_metadata["world_name"]
    world_metadata = get_world_metadata_dict()[world_name]
    return region_metadata, world_metadata


def cast_condition_array_to_int(strs):
    n_rows = len(strs)
    n_cols = len(strs[0])
    assert all(len(row) == n_cols for row in strs)

    arr = np.full((n_rows, n_cols), fill_value=-1, dtype=int)
    for ri, row in enumerate(strs):
        for ci, s in enumerate(row):
            if s == "":
                pass
            else:
                # force it to be parseable as int
                try:
                    v = int(s)
                except ValueError as e:
                    print("condition array must be all ints")
                    raise e
                if v < 0:
                    raise ValueError("condition array must be all ints >= 0")
                arr[ri, ci] = v
    return arr


def get_condition_shorthand_fp(world_name, map_variable):
    root_dir = get_root_dir_for_world(world_name)
    shorthand_filename = f"ImageKey_{map_variable}_condition.csv"
    fp = os.path.join(root_dir, "ControlConditions", shorthand_filename)
    return fp


def get_root_dir_for_world(world_name):
    world_metadata = get_world_metadata_dict()[world_name]
    root_dir = world_metadata["root_dir"]
    return root_dir


def get_root_dir_for_world_of_region(region_name):
    region_metadata, world_metadata = get_region_and_world_metadata(region_name)
    root_dir = world_metadata["root_dir"]
    return root_dir


def get_condition_shorthand_dict(world_name, map_variable):
    shorthand_fp = get_condition_shorthand_fp(world_name, map_variable)

    d = {}
    dict_key_colname = "shorthand"
    with open(shorthand_fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["colors_rgba"] = parse_colors_rgba(row["colors_rgba"])
            row["min"] = int(row["min"]) if row["min"] != "" else None
            row["max"] = int(row["max"]) if row["max"] != "" else None
            assert (row["min"] is None or row["max"] is None) or (row["min"] <= row["max"])
            dict_key = int(row[dict_key_colname])
            assert dict_key not in d
            d[dict_key] = row
    return d


def get_rgb_int_to_condition_dict(world_name, map_variable):
    shorthand_dict = get_condition_shorthand_dict(world_name, map_variable)
    d = {}
    for sh, row in shorthand_dict.items():
        colors_rgba = row["colors_rgba"]
        for tup in colors_rgba:
            r,g,b,a = tup  # just to check that they're 4-tuples
            assert a == 255, a
            n = 1000000*r + 1000*g + b
            assert n not in d, f"duplicate color {n}"
            d[n] = row
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


def get_default_values_of_conditions(world_name, map_variable):
    print(f"getting default values of conditions for {map_variable=} in {world_name=}")
    shorthand_dict = get_condition_shorthand_dict(world_name, map_variable)
    shorthand_to_default_value = {-1: 0}  # undefined condition always defaults to 0 for any variable
    for sh in shorthand_dict:
        assert type(sh) is int and sh >= 0, sh
        min_val = shorthand_dict[sh]["min"]
        max_val = shorthand_dict[sh]["max"]
        default_val = get_default_value_from_min_and_max(min_val, max_val)
        shorthand_to_default_value[sh] = default_val
    return shorthand_to_default_value


def get_default_value_array(region_name, map_variable):
    world_name = get_world_name_from_region_name(region_name)
    default_value_by_condition = get_default_values_of_conditions(world_name, map_variable)
    condition_arr = get_condition_int_array_for_region(region_name, map_variable)
    default_value_arr = translate_array_by_dict(condition_arr, default_value_by_condition)
    return default_value_arr


def get_world_name_from_region_name(region_name):
    return get_region_metadata_dict()[region_name]["world_name"]


def get_default_value_from_min_and_max(min_val, max_val):
    # if condition has min and max, use average
    # if condition has only min or max, use that
    # if condition has neither, use 0 (or NaN? but want to seed from the default for elevation, so use 0)
    assert type(min_val) in [int, type(None)], min_val
    assert type(max_val) in [int, type(None)], max_val
    if min_val is None and max_val is None:
        return 0
    elif min_val is None and max_val is not None:
        return max_val
    elif max_val is None and min_val is not None:
        return min_val
    else:
        return int(round((min_val + max_val)/2))


def create_cada_ii_default_value_dict(map_variable):
    # create dict of icosa point : default elevation value (or other map variable)
    print(f"creating Cada II default value dict for {map_variable}")
    world_name = "Cada II"
    default_value_by_condition = get_default_values_of_conditions(world_name, map_variable)
    return default_value_by_condition


def translate_array_by_dict(arr, d, default_value=None):
    # look up each element in the dict and put its value in the corresponding place in new array
    contains = np.vectorize(d.__contains__)(arr)
    bad_colors = np.unique(arr[~contains])
    if len(bad_colors) > 0:
        print(f"unrecognized colors defaulting to {default_value}:", ", ".join(str(x) for x in bad_colors))
    return np.vectorize(lambda x: d.get(x, default_value))(arr)


def rgb_to_int(arr):
    # represent e.g. (212, 85, 305) as 212085305
    assert len(arr.shape) == 3
    assert arr.shape[-1] == 3
    return arr[:,:,2] + 1000*arr[:,:,1] + 1000000*arr[:,:,0]


def get_rgba_color_array_from_image_array(arr):
    raise Exception("deprecated")
    # # collapse the last dimension into tuples; maybe np has a better way to do this but I can't find it
    # color_arr = np.empty(shape=arr.shape[:-1], dtype=object)  # object is better dtype so np not trying to iterate
    # assert len(color_arr.shape) == 2, color_arr.shape
    # for xi in range(color_arr.shape[0]):
    #     for yi in range(color_arr.shape[1]):
    #         color_arr[xi,yi] = tuple(arr[xi,yi,:])
    # return color_arr


def get_condition_int_array_from_image_array(arr, color_to_int):
    r_size, c_size, rgba_len = arr.shape
    assert rgba_len == 4
    assert (arr[:,:,3] == 255).all(), "don't put transparent stuff in the images"
    # use uint32 to prevent overflow when making the integer representations of colors
    # (max value 255255255, which is about 2**28)
    arr = arr[:,:,:3].astype(np.uint32)  
    arr = rgb_to_int(arr)
    arr = translate_array_by_dict(arr, color_to_int, default_value=-1)
    assert (arr >= -1).all()
    return arr


def get_rc_size_of_image(image_fp):
    im = Image.open(image_fp)
    width, height = im.size
    r_size = height
    c_size = width
    return r_size, c_size


def get_region_template_image_fp(region_name):
    world_name = get_world_name_from_region_name(region_name)
    control_condition_dir = get_control_condition_dir(world_name)
    fname = f"{region_name}_template.png"
    fp = os.path.join(control_condition_dir, fname)
    return fp


def get_lattice_for_region(region_name):
    image_fp = get_region_template_image_fp(region_name)
    latlon00, latlon01, latlon10, latlon11 = get_latlon_dict()[region_name]
    im = Image.open(image_fp)

    shrink_debug = False
    if shrink_debug:
        im = shrink_resolution(im)
    else:
        print("opened image with PIL; check memory usage")

    width, height = im.size
    print(f"image of region {region_name} has shape ({width}, {height}), total of {width*height} pixels")

    image_lattice = LatitudeLongitudeLattice(
        height, width,  # rows, columns
        latlon00, latlon01, latlon10, latlon11
    )
    return image_lattice


def get_color_to_shorthand_int_dict(rgb_int_to_condition_dict):
    return {n: int(row["shorthand"]) for n, row in rgb_int_to_condition_dict.items()}


def get_lattice_and_df_for_region(region_name, map_variable, with_coords=False):
    lattice = get_lattice_for_region(region_name)

    rgb_int_to_condition_dict = get_rgb_int_to_condition_dict(region_name, map_variable)
    color_to_shorthand_int = get_color_to_shorthand_int_dict(rgb_int_to_condition_dict)

    image_fp = get_condition_image_fp(region_name, map_variable)
    im = Image.open(image_fp)
    arr = np.array(im)
    assert arr.shape[-1] == 4, arr.shape  # RGBA dimension
    str_arr = get_condition_int_array_from_image_array(arr, color_to_shorthand_int)

    df = lattice.create_dataframe(with_coords=with_coords)

    df_index = df.index
    point_indices = lattice.get_point_indices()
    assert len(df_index) == len(point_indices)
    assert list(df_index) == list(point_indices)
    df[map_variable] = ["" for p_i in point_indices]
    original_df_size = df.size
    str_lst = []

    print("strings found in image:", np.unique(str_arr))
    print("adding condition values to DataFrame")
    for (r,c), point_number in zip(lattice.get_rc_generator(), point_indices):
        if c == 0:
            print(f"row {r}/{lattice.r_size} ({100*r/lattice.r_size :.2f}%)")
        assert df.size == original_df_size, "df growing out of control"  # catching assignment bug that tries to make new column/multi-index on every assignment
        str_val = str_arr[r][c]
        # df[map_variable][point_number] = str_val  # too slow?
        str_lst.append(str_val)
        # print("str_val is {}".format(str_val))
        # print("df size is now {}".format(df.size))
    df[map_variable] = str_lst

    print("done adding condition values to DataFrame")
    return lattice, df


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
    im = im.resize((new_r, new_c), Image.NEAREST)
    return im


def find_icosa_points_near_image_lattice_points(image_lattice, icosa_point_tolerance, planet_radius):
    print("finding icosa points near image lattice points")
    d = {}
    for r in range(image_lattice.r_size):
        print(f"row {r}/{image_lattice.r_size} ({100*r/image_lattice.r_size :.2f}%)")
        for c in range(image_lattice.c_size):
            point_number = image_lattice.get_point_number_from_lattice_position(r,c)
            point = image_lattice.points[point_number]
            latlon = point.latlondeg()
            nearest_icosa_point, distance_normalized, distance_km = icm.get_nearest_icosa_point_to_latlon(latlon, icosa_point_tolerance, planet_radius)
            # print(f"\nimage ({r},{c}); point #{point_number}; point {point}; icosa point within {icosa_point_tolerance_km} km = {nearest_icosa_point} at distance of {distance_km} km away")
            d[point_number] = nearest_icosa_point
    print("done finding icosa points near image lattice points")
    return d


def get_approx_icosa_point_codes_from_usps(usps, icosa_distance_tolerance_normalized):
    print("getting approx icosa points for USPs")
    approx_icosa_point_codes = []
    for pi, p in enumerate(usps):
        if pi % 1000 == 0 and pi != 0:
            print(f"point {pi}/{len(usps)} ({100*pi/len(usps) :.2f}%)")
        approx_p, d_norm, d_units = icm.get_nearest_icosa_point_to_latlon(p.latlondeg(), icosa_distance_tolerance_normalized, planet_radius=1)
        approx_pc = approx_p.point_code
        approx_icosa_point_codes.append(approx_pc)
    print("done getting approx icosa points for USPs")
    return approx_icosa_point_codes


def write_image_pixel_to_icosa_point_code(region_name, overwrite_existing=False):
    print(f"writing image pixel to icosa point code correspondence for {region_name=}")
    pixel_to_icosa_fp = get_region_metadata_dict()[region_name]["pixel_to_icosa_fp"]

    try:
        existing_rc_to_pc_dict = get_image_pixel_to_icosa_point_code_from_memo(region_name)
    except FileNotFoundError:
        existing_rc_to_pc_dict = {}

    if overwrite_existing:
        if len(existing_rc_to_pc_dict) > 0:
            print(f"Warning: overwriting existing icosa point codes at {pixel_to_icosa_fp}")
        rc_to_point_code = get_image_pixel_to_icosa_point_code_from_calculation(region_name)
    else:
        # fill in the missing pixels, if any
        # I added this here because I originally made a fencepost error and missed the final king on every image
        r_size, c_size = get_rc_size_of_image(region_name)
        missing_pixels = []
        for r in range(r_size):
            for c in range(c_size):
                if (r,c) not in existing_rc_to_pc_dict:
                    missing_pixels.append((r,c))
        remaining_rc_to_point_code = get_image_pixel_to_icosa_point_code_from_calculation(region_name, pixels=missing_pixels)
        print(f"only calculating {len(remaining_rc_to_point_code)} point codes")
        rc_to_point_code = existing_rc_to_pc_dict
        rc_to_point_code.update(remaining_rc_to_point_code)

    # now write the whole array
    lines = []
    for r in range(r_size):
        row_points = []
        for c in range(c_size):
            rc = (r, c)
            p_i = rc_to_point_code[rc]
            row_points.append(p_i)
        row_str = ",".join(str(x) for x in row_points) + "\n"
        lines.append(row_str)
    with open(pixel_to_icosa_fp, "w") as f:
        for line in lines:
            f.write(line)

    print(f"done writing image pixel to icosa point code correspondence for {region_name=}")


def get_point_codes_in_order_from_memo(region_name):
    rc_to_pc = get_image_pixel_to_icosa_point_code_from_memo(region_name)
    r_size, c_size = get_rc_size(rc_to_pc.keys())
    # create flat list
    pcs = []
    for r in range(r_size):
        for c in range(c_size):
            pc = rc_to_pc[(r,c)]
            pcs.append(pc)
    assert len(pcs) == r_size * c_size
    return pcs


def get_condition_int_array_for_region(region_name, map_variable):
    region_metadata = get_region_metadata_dict()[region_name]
    image_fp = get_condition_image_fp(region_name, map_variable)
    world_name = region_metadata["world_name"]
    world_metadata = get_world_metadata_dict()[world_name]

    rgb_int_to_condition_dict = get_rgb_int_to_condition_dict(world_name, map_variable)
    color_to_shorthand_int = get_color_to_shorthand_int_dict(rgb_int_to_condition_dict)
    
    im = Image.open(image_fp)
    arr = np.array(im)
    assert arr.shape[-1] == 4, arr.shape  # RGBA dimension
    int_arr = get_condition_int_array_from_image_array(arr, color_to_shorthand_int)
    return int_arr


def get_all_map_variable_names(world_name):
    # based on what images are in the directory of control conditions
    dirpath = get_control_condition_dir(world_name)
    pattern = "(?P<region_name>[^_]+)_(?P<variable_name>[^_]+)_condition.png"
    matches = [re.match(pattern, fp) for fp in os.listdir(dirpath)]
    vars = sorted(set(match.group("variable_name") for match in matches if match is not None))
    return vars


def get_control_point_dataframe(world_name):
    # make a dataframe with all control conditions for all variables
    region_metadata = get_region_metadata_dict()
    map_variables = get_all_map_variable_names(world_name)
    region_dfs = []
    for region_name in region_metadata.keys():
        region_df = pd.DataFrame(columns=[f"{x}_condition" for x in map_variables], dtype=np.int8)
        pn_arr = get_image_pixel_to_icosa_point_code_from_memo(region_name)
        pns = pn_arr.flatten()
        test_pn = pns[0]
        print(f"{test_pn=}")
        for var in map_variables:
            colname = f"{var}_condition"
            condition_arr = get_condition_int_array_for_region(region_name, var)
            assert condition_arr.shape == pn_arr.shape
            print(f"adding {colname} in region {region_name} to DataFrame")
            conditions = condition_arr.flatten()
            max_condition = conditions.max()
            assert max_condition <= 2**7 - 1, "too many conditions to store as np.int8 (we need signed to store -1)"
            conditions = conditions.astype(np.int8)
            s = pd.Series(dict(zip(pns, conditions)), name=colname)
            region_df[colname] = s
            assert list(region_df.index) == list(s.index)
            assert test_pn in region_df.index
        region_dfs.append(region_df)
    df = pd.concat(region_dfs)
    df["pc"] = icm.get_point_codes_from_point_numbers(df.index)
    print("done getting control point dataframe")
    return df


def plot_default_values_by_region(world_name, map_variable):
    region_metadata = get_region_metadata_dict()
    for region_name in region_metadata.keys():
        default_value_arr = get_default_value_array(region_name, map_variable)
        plt.imshow(default_value_arr)
        plt.colorbar()
        plt.show()


def plot_icosa_point_number_to_value_dict(pn_to_val):
    print("plotting icosa point number to value dict")
    pns = sorted(pn_to_val.keys())
    pns = random.sample(pns, min(len(pns), 10000))  # debug
    latlons = icm.get_latlons_from_point_numbers(pns)
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
    world_name = "Cada II"
    region_metadata = get_region_metadata_dict()
    world_metadata = get_world_metadata_dict()["Cada II"]
    map_variable = "elevation"

    # plot_default_values_by_region(world_name, map_variable)
    df = get_control_point_dataframe(world_name)
    df.to_hdf("control_data.h5", key="control_data")

    # for region_name in region_metadata.keys():
    #     # write_image_conditions_as_image_shape_in_shorthand(region_name, map_variable)
    #     arr = get_condition_int_array_for_region(region_name, map_variable)
    #     plt.imshow(arr)
    #     plt.show()

    # image_lattice, df = get_lattice_and_df_from_image(image_fp, latlon00, latlon01, latlon10, latlon11, color_to_str, map_variable)
    # icosa_points = find_icosa_points_near_image_lattice_points(image_lattice, icosa_point_tolerance_km, planet_radius_km)

    # icosa_iterations_of_precision = IcosahedronMath.get_iterations_needed_for_edge_length(icosa_point_tolerance_km, planet_radius_km)
    # n_icosa_points = IcosahedronMath.get_points_from_iterations(icosa_iterations_of_precision)

    # # now get map lattice points which are inside the image lattice
    # image_lattice_points_in_order = image_lattice.points
    # point_values_to_assign = {}
    # t0 = time.time()
    # for p_i in range(n_icosa_points):
    #     if p_i % 1000 == 0 and p_i != 0:
    #         progress_proportion = p_i / n_icosa_points
    #         elapsed = time.time() - t0
    #         rate = progress_proportion / elapsed
    #         remaining_proportion = 1 - progress_proportion
    #         eta = remaining_proportion / rate
    #         eta_str = str(datetime.timedelta(seconds=eta))
    #         print("icosa point {} of {} ({}%), ETA {}".format(p_i, n_icosa_points, 100*progress_proportion, eta_str))
    #     usp = IcosahedronMath.get_usp_from_point_number(p_i)
    #     in_image = image_lattice.contains_point_latlon(usp)
    #     if in_image:
    #         closest_image_point_index, closest_image_point = image_lattice.closest_point_to(usp)
    #         value_to_assign = df[map_variable][closest_image_point_index]
    #         point_values_to_assign[p_i] = value_to_assign
    #         # print("point {} now has value {}".format(p_i, value_to_assign))

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
