from IcosahedronPointDatabase import IcosahedronPointDatabase
import IcosahedronPointDatabase as icdb
import IcosahedronMath as icm
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm
from GreatCircleDistanceMatrix import GreatCircleDistanceMatrix
from BiDict import BiDict
import LoadMapData
import PlottingUtil as pu
import FindPointsInCircle as find

import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree



def get_point_number_cache_fp(region_center_latlondeg, region_radius_great_circle_km):
    return f"PointNumberCache_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km.txt"


def get_point_numbers_in_region_from_db(db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    point_numbers_in_db = db.get_all_point_numbers_with_data()
    print(f"checking {len(point_numbers_in_db)} points")

    # try different algorithms for finding the correct set of points
    filter_point_numbers_in_region = find.filter_point_codes_in_region_one_by_one
    # filter_point_numbers_in_region = filter_point_numbers_in_region_all_at_once

    point_numbers_in_region_in_db = filter_point_numbers_in_region(point_numbers_in_db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km)

    # in case of crash
    print("---- point_numbers_in_region_in_db ----")
    print(point_numbers_in_region_in_db)
    print("//// point_numbers_in_region_in_db ////")

    write_point_numbers_to_cache(point_numbers_in_region_in_db, region_center_latlondeg, region_radius_great_circle_km)
    return point_numbers_in_region_in_db


def write_point_numbers_to_cache(point_numbers, region_center_latlondeg, region_radius_great_circle_km):
    raise Exception("deprecated")
    # point_number_cache_fp = get_point_number_cache_fp(region_center_latlondeg, region_radius_great_circle_km)
    # # keep everything that's already there
    # try:
    #     with open(point_number_cache_fp) as f:
    #         lines = f.readlines()
    # except FileNotFoundError:
    #     lines = []
    # existing_pns = set(int(l.strip()) for l in lines)
    # point_numbers = set(point_numbers)
    # if len(point_numbers - existing_pns) == 0:
    #     print("all points are already cached")
    #     return

    # point_numbers |= existing_pns
    # with open(point_number_cache_fp, "w") as f:
    #     for pn in sorted(point_numbers):
    #         f.write(f"{pn}\n")


def read_point_numbers_from_cache(region_center_latlondeg, region_radius_great_circle_km):
    raise Exception("deprecated")
    # point_number_cache_fp = get_point_number_cache_fp(region_center_latlondeg, region_radius_great_circle_km)
    # print(f"looking for {point_number_cache_fp}")
    # if os.path.exists(point_number_cache_fp):
    #     print(f"reading from cache file")
    #     with open(point_number_cache_fp) as f:
    #         lines = f.readlines()
    #     point_numbers = [int(l.strip()) for l in lines]
    #     return point_numbers
    # else:
    #     print("cache file not found")
    #     raise FileNotFoundError(point_number_cache_fp)


def get_point_numbers_with_data_in_region(db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    print("getting point numbers with data in region")
    try:
        res = read_point_numbers_from_cache(region_center_latlondeg, region_radius_great_circle_km)
        print("got point numbers from cache")
    except FileNotFoundError:
        print("calculating point numbers with data in region using icosa math")
        point_numbers = list(get_point_numbers_in_region_from_db(db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km))
        res = point_numbers
    print("-- done getting point numbers with data in region")
    return res


def get_points_in_region_old(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations):
    raise Exception("deprecated")
    # print("getting points in region")
    # try:
    #     res = get_points_in_region_from_file(region_center_latlondeg, region_radius_great_circle_km, iterations)
    #     print("got points from file")
    # except FileNotFoundError:
    #     print("calculating points in region using icosa math")
    #     points = get_points_in_region_raw(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations)
    #     fp = get_points_at_resolution_cache_fp(region_center_latlondeg, region_radius_great_circle_km, iterations)
    #     with open(fp, "w") as f:
    #         f.write("\n".join(str(pn) for pn in points))
    #     res = points
    # print("-- done getting points in region")
    # return res


def get_points_at_resolution_cache_fp(region_center_latlondeg, region_radius_great_circle_km, iterations):
    return f"PointsAtResolution_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km_iterations_{iterations}.txt"


def get_points_in_region_from_file(region_center_latlondeg, region_radius_great_circle_km, iterations):
    raise Exception("deprecated")
    # fp = get_points_at_resolution_cache_fp(region_center_latlondeg, region_radius_great_circle_km, iterations)
    # print(f"looking for {fp}")
    # if os.path.exists(fp):
    #     print(f"reading from cache file")
    #     with open(fp) as f:
    #         lines = f.readlines()
    #         return [int(l.strip()) for l in lines]
    # else:
    #     print("cache file not found")
    #     raise FileNotFoundError(fp)


def get_points_in_region(center_pc, region_radius_gc, max_iterations):
    return icm.get_region_around_point_code_by_spreading(center_pc, region_radius_gc, max_iterations)

    # old
    # procedure: start with icosa starting points
    # have function which tells you the farthest a point's descendants can get from it
    # calculate the point's distance from the region center
    # if that distance - max_distance_of_descendant is still too far away, then throw this point out and don't bother looking at its descendants

    ## region_center_xyz = mcm.unit_vector_lat_lon_to_cartesian(*region_center_latlondeg, deg=True)
    # distance = lambda pn: icm.get_distance_point_number_to_xyz_great_circle(pn, region_center_xyz, radius=planet_radius_km)
    # points_in_region = []
    # points_whose_children_could_be_in_region = []

    # # for the poles, just check whether they're in the region or not, since they don't have descendants
    # for pn in range(2):
    #     # print(f"checking point {pn}")
    #     d = distance(pn)
    #     # print(f"distance from {pn} to region center is {d}")
    #     if d < region_radius_great_circle_km:
    #         # print(f"{pn} is in the region")
    #         points_in_region.append(pn)

    # starting_points = list(range(2, 12))
    # to_check = starting_points
    # for iteration in range(0, iterations+1):
    #     n_to_check = len(to_check)
    #     if n_to_check == 0:
    #         break
    #     to_check_next_round = []
    #     print(f"checking iteration {iteration}")
    #     # for each point, only check its actual distance on the first time you see it
    #     for i, pn in enumerate(to_check):
    #         if i % 100 == 0 and i != 0:
    #             print(f"{i}/{n_to_check} this round (iteration {iteration})")
    #         # print(f"checking point {pn}")
    #         if iteration == icm.get_iteration_born_from_point_number(pn):
    #             # check its distance, put it in points_in_region if it fits
    #             d = distance(pn)
    #             # print(f"distance from {pn} to region center is {d}")
    #             if d < region_radius_great_circle_km:
    #                 # print(f"{pn} is in the region")
    #                 points_in_region.append(pn)
    #         else:
    #             # print(f"already seen point {pn} by iteration {iteration} because it was born at iteration {icm.get_iteration_born(pn)}")
    #             pass

    #         # now, if its children *starting after this iteration* can ever get into the region, keep it in to_check and add the children as well, else throw it out

    #         iteration_of_next_child = iteration + 1
    #         should_check_descendants = descendants_of_point_can_ever_be_in_region(pn, region_center_xyz, region_radius_great_circle_km, planet_radius_km, iteration_of_next_child)
    #         if should_check_descendants:
    #             # print("should check descendants")
    #             to_check_next_round.append(pn)
    #             children = icm.get_children_from_point_number(pn, iteration+1)
    #             # print(f"children: {children}")
    #             to_check_next_round += children

    #     to_check = to_check_next_round

    # return points_in_region


def descendants_of_point_can_ever_be_in_region(pn, region_center_xyz, region_radius_great_circle_km, planet_radius_km, iteration_of_next_child):
    # print(f"checking descendant min distance to region center, pn={pn}, iteration_of_next_child={iteration_of_next_child}")
    farthest_distance_descendant_can_be_from_pn = icm.get_farthest_distance_descendant_can_be(pn, radius=planet_radius_km, iteration_of_next_child=iteration_of_next_child)
    # print(f"farthest distance of descendant from pn: {farthest_distance_descendant_can_be_from_pn}")
    current_distance = icm.get_distance_point_number_to_xyz_great_circle(pn, region_center_xyz, radius=planet_radius_km)
    # print(f"current distance from pn to region center: {current_distance}")
    closest_descendant_can_be_to_region_center = max(0, current_distance - farthest_distance_descendant_can_be_from_pn)
    # print(f"closest descendant can be to region center: {closest_descendant_can_be_to_region_center}")
    return closest_descendant_can_be_to_region_center <= region_radius_great_circle_km


def interpolate_at_points_nearest_neighbor(pcs_to_interpolate_at, pcs_to_interpolate_from, variable_name, db, max_nn_distance=None):
    # interpolate the conditions at the points_at_this_resolution
    print("interpolating nearest neighbor")
    known_values = db.get_dict(pcs_to_interpolate_from, variable_name)
    if len(set(pcs_to_interpolate_at) - set(pcs_to_interpolate_from)) == 0:
        # already know values of all these points, don't bother getting xyz or doing nearest neighbor calculation
        return {pc: known_values[pc] for pc in pcs_to_interpolate_at}

    nn_pc_lookup, d_lookup = icm.get_nearest_neighbors_pc_to_pc_with_distance(query_pcs=pcs_to_interpolate_at, candidate_pcs=pcs_to_interpolate_from, k_neighbors=1, allow_self=False)
    interpolated = pd.Series(dtype=int)
    for i, pc in enumerate(pcs_to_interpolate_at):
        if i % 100 == 0:
            print(f"interpolating at points; progress {i}/{len(pcs_to_interpolate_at)}")
        if pc in known_values: # and known_values[pc] != -1:
            # we already know its condition, no need to do nearest neighbors
            interpolated[pc] = known_values[pc]
            # print(interpolated)
            # print("a")
        else:
            ds = d_lookup[pc]
            nn_pcs = nn_pc_lookup[pc]
            assert len(ds) == 1
            assert len(nn_pcs) == 1
            d = ds[0]
            nn_pc = nn_pcs[0]

            assert d != 0 and nn_pc != pc, "shouldn't use point itself as a nearest neighbor; either take its value if specified, or take a different nearby value if this point is unspecified"
            if max_nn_distance is None or d <= max_nn_distance:
                el_cond = known_values[nn_pc]
                assert type(el_cond) is int, type(el_cond)
                interpolated[pc] = el_cond
                print(f"placing interpolated value of {el_cond} at {pc}")
            else:
                # don't interpolate here, the nearest neighbor is too far away
                print(f"placing undetermined value at {pc}")
                interpolated[pc] = -1
    print("-- done interpolating nearest neighbor")
    
    # debug
    # print(f"interpolated result of type {type(interpolated)}:")
    # print(interpolated)
    # input("check")

    return interpolated


def interpolate_conditions(db, pcs_with_data_in_region, pcs_in_region_at_resolution, planet_radius_km, elevation_condition_to_default_value):
    # interpolate condition at other points as nearest neighbor
    # (with some max distance to that neighbor so we don't get things like
    # the middle of the ocean thinking it has to be a coast/shallow
    # because that's what's on the edge of the nearest image thousands of km away)

    # using the points in the region with data as interpolation, we will generate elevations at the points_at_this_resolution AND the points that already have data
    points_to_interpolate_at = list(set(pcs_in_region_at_resolution) | set(pcs_with_data_in_region))
    interpolated_el_conds = interpolate_at_points_nearest_neighbor(
        pcs_to_interpolate_at=points_to_interpolate_at,
        pcs_to_interpolate_from=pcs_with_data_in_region,
        variable_name="elevation_condition",
        db=db,
        max_nn_distance=100/planet_radius_km,
    )
    # write these to the db
    print(f"got interpolated elevation conditions at {len(interpolated_el_conds)} points")
    print(interpolated_el_conds)
    pcs = interpolated_el_conds.index
    old_el_conds = db.get_series(pcs, "elevation_condition")
    old_na_mask = old_el_conds == -1
    new_na_mask = interpolated_el_conds == -1
    
    # if the old condition was NA (-1), then we don't care what the new one is
    # if the old condition was not NA, the new condition must be the same as it
    check_new_el_conds = interpolated_el_conds[~old_na_mask]
    check_old_el_conds = old_el_conds[~old_na_mask]
    equal_mask = check_new_el_conds == check_old_el_conds
    if (~equal_mask).sum() > 0:
        # some points changed but should have kept their old condition value
        # tell user where this happened
        unequal_df = pd.DataFrame([check_old_el_conds[~equal_mask], check_new_el_conds[~equal_mask]], columns=["old", "new"])
        print("elevation conditions changed during interpolation:")
        print(unequal_df)
        raise RuntimeError("elevation conditions changed but shouldn't have; see DataFrame above")

    pu.plot_variable_scattered_from_dict(interpolated_el_conds)
    if input("commit these results to db in RAM? y/n (default n)") == "y":
        db[interpolated_el_conds.index, "elevation_condition"] = interpolated_el_conds
        default_els = [elevation_condition_to_default_value[el_cond] for el_cond in interpolated_el_conds]
        db[interpolated_el_conds.index, "elevation"] = default_els
    if input("write these results to file? y/n (default n)") == "y":
        db.write_hdf()
    points_to_edit = points_to_interpolate_at
    return points_to_edit


if __name__ == "__main__":
    db_root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"
    db = IcosahedronPointDatabase.load(db_root_dir)
    df = db.df
    planet_radius_km = icm.CADA_II_RADIUS_KM

    ## region_center_latlondeg, region_radius_great_circle_km = (10, -87), 1000  # Western Amphoto
    ## region_center_latlondeg, region_radius_great_circle_km = (-87, 10), 1000  # somewhere in O-Z because I originally mixed up latlon
    ## region_center_latlondeg, region_radius_great_circle_km = (90, 0), 2000  # North Pole
    ## region_center_latlondeg, region_radius_great_circle_km = (-14, -115), 2000  # Thiuy-Rainia Bay
    ## region_center_latlondeg, region_radius_great_circle_km = (86.5, -13), 250  # small region in Tomar Strait in Mienta, for testing on smaller regions
    ## region_center_latlondeg, region_radius_great_circle_km = (25, -84), 2000  # Jhorju
    ## region_center_latlondeg, region_radius_great_circle_km = (-54.28119589256169, 175.64265081464623), 250  # random from 2022-07-16
    ## region_center_latlondeg, region_radius_great_circle_km = (26.083351229768834, 94.04570559120195), 2000  # northern Mienta, from a random point
    ## region_center_point_code = icm.get_nearest_icosa_point_to_latlon(region_center_latlondeg, maximum_distance=1, planet_radius=icm.CADA_II_RADIUS_KM)

    # to choose random one
    ## region_center_point_code = icm.get_random_point_code(min_iterations=3, expected_iterations=6)
    ## region_center_latlondeg = icm.get_latlon_from_point_code(region_center_point_code)
    ## region_radius_great_circle_km = 2000
    ## region_center_latlondeg, region_radius_great_circle_km = (
    #    UnitSpherePoint.get_random_unit_sphere_point().latlondeg(), 250
    # )

    # to choose existing point file (they are just lists of the point codes in a given area)
    pc_dir = "PointFiles"
    # fname, fp = icdb.get_random_point_code_file(pc_dir)
    # fname = "pcs_in_db_2022-11-22_K003201212211_0.02438586135795968.txt"  # small region
    # fname = "pcs_in_db_2022-11-22_D030022203113_0.11116449190061897.txt"  # region on coast of Oligra with a good amount of both land and sea
    # fname = "pcs_in_db_2022-11-22_K03302310332_0.1443687630207259.txt"
    # fp = os.path.join(pc_dir, fname)
    # db.verify_points_exist(pcs_with_data_in_region)
    ## region_center_pc, region_radius_gc = icdb.get_point_code_and_distance_from_filename(fname)

    region_center_pc, region_radius_gc = "D030022203113", 0.11116449190061897
    ## region_center_pc, region_radius_gc = "B", 0.01
    ## region_center_pc, region_radius_gc = "D3", 0.05  # all-land area of Oligra, for testing condition interpolation

    pc_fp = icdb.get_point_code_filename(region_center_pc, region_radius_gc, prefix_str="pcs_in_db_2022-12-16", parent_dir=pc_dir)
    if not os.path.exists(pc_fp):
        icdb.make_point_code_file_for_region(db, region_center_pc, region_radius_gc)
    pcs_with_data_in_region = icdb.get_point_codes_from_file(pc_fp)

    region_radius_gc_km = region_radius_gc * planet_radius_km
    center_latlondeg = icm.get_latlon_from_point_code(region_center_pc)

    print(f"region centered at {region_center_pc} {center_latlondeg} deg with radius {region_radius_gc_km} km")
    # pu.scatter_icosa_points_by_code(pcs_with_data_in_region)

    power_law_param = 0.25  # 1 is uniform dist, >1 is more weight toward 1 and less toward 0, a=0 is all weight at 0, a=inf is all weight at 1
    power_law = lambda: np.random.power(power_law_param)
    circle_radius_dist = lambda: power_law()
    el_stdev = 15
    resolution_iterations = 9
    n_circles = 10000
    interpolate_condition_variables = False  # turn this on if you're doing a new region so it can get the elevation conditions at all the points you want
    pcs_in_region_at_resolution = icm.get_region_around_point_code_by_spreading(region_center_pc, region_radius_gc, resolution_iterations)

    # add missing points to the dataframe
    missing_pcs = [x for x in pcs_in_region_at_resolution if x not in df.index]
    missing_df = pd.DataFrame(index=missing_pcs, columns=df.columns)
    condition_variables = db.get_condition_variables()
    value_variables = db.get_value_variables()
    missing_df.loc[:, condition_variables] = -1
    missing_df.loc[:, value_variables] = 0
    db.df = pd.concat([df, missing_df]).sort_index()

    # use this to check if the point locations look right
    # (is it actually interpolating conditions onto icosa lattice points, for instance?
    # (which should locally look like a triangular/hexagonal lattice
    # with random interloping image pixel points)
    # or is it just taking the image pixels?
    # (which should locally look like a rectangular lattice))
    # pu.plot_variables_scattered(db, pcs_with_data_in_region, ["elevation_condition", "elevation"])

    # pu.plot_variable_interpolated(db, points_with_data_in_region, "elevation", resolution=1000)
    # input("press enter to continue")

    # edit the region and then plot again

    condition_shorthand_dict = LoadMapData.get_condition_shorthand_dict(world_name="Cada II", map_variable="elevation")
    elevation_condition_to_default_value = LoadMapData.get_default_values_of_conditions(world_name="Cada II", map_variable="elevation")
    elevation_condition_to_min_value = {sh: condition_shorthand_dict[sh]["min"] for sh in condition_shorthand_dict}
    elevation_condition_to_max_value = {sh: condition_shorthand_dict[sh]["max"] for sh in condition_shorthand_dict}

    if interpolate_condition_variables:
        points_to_edit = interpolate_conditions(db, pcs_with_data_in_region, pcs_in_region_at_resolution, planet_radius_km, elevation_condition_to_default_value)
    else:
        points_to_edit = list(pcs_in_region_at_resolution)
    
    # start by generating random elevation circles in the region
    xyz_array = icm.get_xyz_array_from_point_codes(points_to_edit)
    xyz_tuples = [tuple(xyz) for xyz in xyz_array]
    xyz_dict = BiDict.from_dict(dict(zip(points_to_edit, xyz_tuples)))
    matrix = GreatCircleDistanceMatrix(xyz_array, radius=1)

    # db.add_variable("elevation")
    
    elevation_conditions = db.get_dict(points_to_edit, "elevation_condition")

    n_passed = 0
    n_failed = 0
    for c_i in range(n_circles):
        print(f"circle {c_i} / {n_circles}")
        circle_radius_gc = circle_radius_dist()
        pc_center = random.choice(points_to_edit)
        xyz_center = np.array(xyz_dict[pc_center])
        distances = matrix.get_distances_to_point(xyz_center)
        p_xyzs_in_circle = matrix.get_points_within_distance_of_point(xyz_center, circle_radius_gc)
        pcs_in_circle = [xyz_dict[p_xyz] for p_xyz in p_xyzs_in_circle.keys()]
        # pcs_in_circle = icm.get_region_around_point_code_by_spreading(pc_center, circle_radius_gc, resolution_iterations)
        # print(f"this circle contains {pcs_in_circle}")

        d_el = int(round(np.random.normal(0, el_stdev)))
        old_els = db.get_dict(pcs_in_circle, "elevation")
        # print("old_els:", old_els)
        # print(f"values in old elevations: {sorted(set(old_els.values()))}")  # debugging when it is overwriting existing data with default elevations

        # keep track of how much each point can move up or down from where it is right now, and adjust d_el toward 0 (but keep its direction) so that the conditions are all still met (unless that change becomes zero in which case just start over)
        # just look at absolute value of changes that are in the same direction as d_el
        is_rise = d_el >= 0
        d_el = abs(d_el)
        max_move = abs(d_el)
        for pc in pcs_in_circle:
            el = old_els[pc]
            assert el % 1 == 0, f"{el} of type {type(el)}"
            # print(f"point {pn} has old_el {el}")
            el_cond = elevation_conditions[pc]
            if is_rise:
                # check if we would go over the max
                max_val = elevation_condition_to_max_value.get(el_cond)
                if max_val is not None:
                    assert max_val >= el, f"invalid elevation found: {el} exceeds max value of {max_val} at point {pc}"
                    assert type(max_val) is int, el_cond
                    move_size = abs(max_val - el)
                    d_el = min(d_el, move_size)
            else:
                min_val = elevation_condition_to_min_value.get(el_cond)
                if min_val is not None:
                    assert el >= min_val, f"invalid elevation found: {el} is below min value of {min_val} at point {pc}"
                    assert type(min_val) is int, el_cond
                    move_size = abs(min_val - el)
                    d_el = min(d_el, move_size)
            if d_el == 0:
                print("d_el reached zero, skipping")
                break
        if d_el == 0:
            n_failed += 1
            continue
        else:
            n_passed += 1

        if not is_rise:
            # convert it back to a fall after minimizing the abs
            d_el = -1 * d_el
        assert d_el % 1 == 0, f"{d_el} of type {type(d_el)}"

        new_els = {pc: old_els[pc] + d_el for pc in pcs_in_circle}
        # check new_els still meet elevation conditions
        all_meet_conditions = True
        for pc in pcs_in_circle:
            el_cond = elevation_conditions[pc]
            max_val = elevation_condition_to_max_value.get(el_cond)
            min_val = elevation_condition_to_min_value.get(el_cond)
            new_val = new_els[pc]
            meets_condition = True
            if min_val is not None:
                meets_condition = meets_condition and min_val <= new_val
            if max_val is not None:
                meets_condition = meets_condition and new_val <= max_val
            if not meets_condition:
                all_meet_conditions = False
                break
            # can be smarter about how we choose d_el based on the most any point here can move up/down
        if all_meet_conditions:
            print("conditions passed, adding to db")
            db.add_value(pcs_in_circle, "elevation", d_el)
            # print(f"added {d_el}")
            # print("new values:", db[pcs_in_circle, "elevation"])
        else:
            print("conditions failed, making new circle")
            raise Exception("this shouldn't happen anymore")

    assert n_passed + n_failed == n_circles
    print(f"condition pass rate {n_passed / n_circles}")

    # pu.plot_variables_scattered_from_db(db, points_to_edit, ["elevation_condition", "elevation"])
    pu.plot_variables_interpolated_from_db(db, points_to_edit, ["elevation_condition", "elevation"], resolution=1000, show=True)
    if input("write these results? y/n (default n)") == "y":
        db.write_hdf()

