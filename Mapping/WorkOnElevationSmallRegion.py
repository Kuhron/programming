import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree
from datetime import datetime

from IcosahedronPointDatabase import IcosahedronPointDatabase
import IcosahedronPointDatabase as icdb
import IcosahedronMath as icm
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm
from GreatCircleDistanceMatrix import GreatCircleDistanceMatrix
from BiDict import BiDict
import LoadMapData as lmd
import PlottingUtil as pu
import FindPointsInCircle as find
from XyzLookupAncestryGraph import XyzLookupAncestryGraph


class InterpolationError(Exception):
    pass


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

    return point_numbers_in_region_in_db


def get_points_at_resolution_cache_fp(region_center_latlondeg, region_radius_great_circle_km, iterations):
    return f"PointsAtResolution_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km_iterations_{iterations}.txt"


def get_points_in_region(center_pc, region_radius_gc, max_iterations):
    return icm.get_region_around_point_code_by_spreading(center_pc, region_radius_gc, max_iterations)


def descendants_of_point_can_ever_be_in_region(pn, region_center_xyz, region_radius_great_circle_km, planet_radius_km, iteration_of_next_child):
    # print(f"checking descendant min distance to region center, pn={pn}, iteration_of_next_child={iteration_of_next_child}")
    farthest_distance_descendant_can_be_from_pn = icm.get_farthest_distance_descendant_can_be(pn, radius=planet_radius_km, iteration_of_next_child=iteration_of_next_child)
    # print(f"farthest distance of descendant from pn: {farthest_distance_descendant_can_be_from_pn}")
    current_distance = icm.get_distance_point_number_to_xyz_great_circle(pn, region_center_xyz, radius=planet_radius_km)
    # print(f"current distance from pn to region center: {current_distance}")
    closest_descendant_can_be_to_region_center = max(0, current_distance - farthest_distance_descendant_can_be_from_pn)
    # print(f"closest descendant can be to region center: {closest_descendant_can_be_to_region_center}")
    return closest_descendant_can_be_to_region_center <= region_radius_great_circle_km


def interpolate_at_points_nearest_neighbor(lns_to_interpolate_at, lns_to_interpolate_from, variable_name, db, xyzg, max_nn_distance=None):
    # interpolate the conditions at the points_at_this_resolution

    interpolated = pd.Series(dtype=int)
    if len(lns_to_interpolate_from) == 0:
        print("no points to interpolate from; returning condition -1 for all query points")
        # no point getting coordinates, etc.
        for ln in lns_to_interpolate_at:
            interpolated.loc[ln] = -1
        return interpolated

    print("interpolating nearest neighbor")
    print("getting known values")
    known_values = db.get_dict(lns_to_interpolate_from, variable_name)
    print("-- done getting known values")
    if len(set(lns_to_interpolate_at) - set(lns_to_interpolate_from)) == 0:
        # already know values of all these points, don't bother getting xyz or doing nearest neighbor calculation
        return pd.Series({ln: known_values[ln] for ln in lns_to_interpolate_at}, dtype=int)

    # db.validate()  # debug
    nn_pc_lookup, d_lookup = icm.get_nearest_neighbors_ln_to_ln_with_distance(query_lns=lns_to_interpolate_at, candidate_lns=lns_to_interpolate_from, xyzg=xyzg, k_neighbors=1, allow_self=False)
    
    for i, ln in enumerate(lns_to_interpolate_at):
        if i % 1000 == 0:
            print(f"interpolating at points; progress {i}/{len(lns_to_interpolate_at)}")
        if ln in known_values: # and known_values[pc] != -1:
            # we already know its condition, no need to do nearest neighbors
            interpolated.loc[ln] = known_values[ln]
            # print(interpolated)
            # print("a")
        else:
            ds = d_lookup[ln]
            nn_pcs = nn_pc_lookup[ln]
            assert len(ds) == 1
            assert len(nn_pcs) == 1
            d = ds[0]
            nn_pc = nn_pcs[0]

            assert d != 0 and nn_pc != ln, "shouldn't use point itself as a nearest neighbor; either take its value if specified, or take a different nearby value if this point is unspecified"
            if max_nn_distance is None or d <= max_nn_distance:
                el_cond = known_values[nn_pc]
                assert type(el_cond) is int, type(el_cond)
                interpolated.loc[ln] = el_cond
                # print(f"placing interpolated value of {el_cond} at {pc}")
            else:
                # don't interpolate here, the nearest neighbor is too far away
                # print(f"placing undetermined value at {pc}")
                interpolated.loc[ln] = -1
    print("-- done interpolating nearest neighbor")
    
    # debug
    # print(f"interpolated result of type {type(interpolated)}:")
    # print(interpolated)
    # input("check")

    return interpolated


def interpolate_conditions(db, lns_with_data_in_region, lns_in_region_at_resolution, planet_radius_km, elevation_condition_to_default_value, xyzg):
    # interpolate condition at other points as nearest neighbor
    # (with some max distance to that neighbor so we don't get things like
    # the middle of the ocean thinking it has to be a coast/shallow
    # because that's what's on the edge of the nearest image thousands of km away)

    # db.validate()  # debug
    # using the points in the region with data as interpolation, we will generate elevations at the points_at_this_resolution AND the points that already have data

    # debug
    # print("plotting elevation before interpolating conditions")
    # pu.plot_variable_at_point_codes(points_to_interpolate_at, db, "elevation", show=True)

    # interpolate only from points that have a defined condition
    existing_el_conds = db.get_series(lns_with_data_in_region, "elevation_condition")
    existing_els = db.get_series(lns_with_data_in_region, "elevation")
    # but also want to make sure we are up to date with the images, TODO read conditions from images
    existing_el_cond_is_defined_mask = existing_el_conds != -1
    lns_with_data_but_undefined_el_cond = lns_with_data_in_region[~existing_el_cond_is_defined_mask]
    lns_to_interpolate_at = list(set(lns_in_region_at_resolution) | set(lns_with_data_but_undefined_el_cond))

    # if we've already generated data here,
    # don't overwrite it with default just because the condition was -1,
    # this means the condition is supposed to be -1
    # (it underwent nearest-neighbor check already) so don't change what's already there
    existing_el_is_defined_mask = existing_els != 0
    print(f"there are {existing_el_is_defined_mask.sum()} points with elevation already")

    # there will still be some false positives where condition is known to be -1
    # (has already been checked for nearest neighbors)
    # but by coincidence the elevation value is also still 0,
    # nothing much to do about this I think
    interpolate_from_mask = existing_el_cond_is_defined_mask | existing_el_is_defined_mask
    lns_to_interpolate_from = existing_el_conds.index[interpolate_from_mask]

    # db.validate()  # debug
    interpolated_el_conds = interpolate_at_points_nearest_neighbor(
        lns_to_interpolate_at=lns_to_interpolate_at,
        lns_to_interpolate_from=lns_to_interpolate_from,
        variable_name="elevation_condition",
        db=db,
        xyzg=xyzg,
        max_nn_distance=100/planet_radius_km,
    )

    # db.validate()  # debug
    print(f"got interpolated elevation conditions at {len(interpolated_el_conds)} points")
    print("interpolated_el_conds:\n", interpolated_el_conds)

    interpolated_el_cond_lns = interpolated_el_conds.index
    old_el_conds = db.get_series(interpolated_el_cond_lns, "elevation_condition")
    # db.validate()  # debug
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
        print("pre-existing elevation conditions changed during interpolation:")
        print(unequal_df)
        raise InterpolationError("pre-existing elevation conditions changed but shouldn't have; see DataFrame above")
    
    # only change anything at points where the elevation condition didn't used to be defined but now is
    change_point_mask = old_na_mask & ~new_na_mask
    change_point_lns = interpolated_el_cond_lns[change_point_mask]
    # print("change_points:", change_points)
    if len(change_point_lns) == 0:
        print("no points were changed by interpolation (thus not writing db to file)")
    else:
        print(f"{len(change_point_lns)} points have had their elevation condition inferred by interpolation")
        el_conds_at_change_points = interpolated_el_conds[change_point_mask]
        # print("el_conds_at_change_points:\n", el_conds_at_change_points)
        els_at_change_points = lmd.translate_array_by_dict(el_conds_at_change_points, elevation_condition_to_default_value)
        # print("els_at_change_points:\n", els_at_change_points)

        # debug
        # print("plotting elevation_condition after interpolating conditions")
        # pu.plot_variable_scattered_from_dict(interpolated_el_conds, title="elevation_condition")

        if True: #input("commit these results to db in RAM? y/n (default n)") == "y":
            db[change_point_lns, "elevation_condition"] = el_conds_at_change_points
            # db.validate()   # debug
            db[change_point_lns, "elevation"] = els_at_change_points
            # db.validate()  # debug

            # debug
            # print("plotting elevation after interpolating conditions")
            # new_els_at_all_points = db.get_series(interpolated_el_conds.index, "elevation")
            # pu.plot_variable_scattered_from_dict(new_els_at_all_points, title="elevation")  # so I can see what is already there

        if True: #input("write these results to file? y/n (default n)") == "y":
            db.write_hdf()
    
    return lns_to_interpolate_at  # we didn't change anything else, so only need to send the points we interpolated to


def run_region_generation(db, planet_radius_km, xyzg):
    df = db.df
    desired_point_prefix = ""
    power_law_param = 2**random.uniform(-2, 2)  # 1 is uniform dist, >1 is more weight toward 1 and less toward 0, a=0 is all weight at 0, a=inf is all weight at 1
    power_law = lambda: np.random.power(power_law_param)  # will be a fraction of region radius
    el_stdev = np.random.uniform(5, 25)
    resolution_iterations = 9
    n_circles = 4000

    # to choose random one
    ## region_center_pc = icm.get_random_point_code(min_iterations=6, expected_iterations=9, max_iterations=9, prefix=desired_point_prefix)
    region_radius_gc = random.randint(800, 1500)/10000
    ## region_radius_gc = 0.01
    ## region_center_latlondeg = icm.get_latlon_from_point_code(region_center_point_code)
    ## region_center_latlondeg, region_radius_great_circle_km = (
    #    UnitSpherePoint.get_random_unit_sphere_point().latlondeg(), 250
    # )

    # to choose based on some condition
    # lns_ocean = df.index[df["elevation_condition"] == 0]
    lns_coast = df.index[df["elevation_condition"] == 1]
    # lns_land = df.index[df["elevation_condition"] == 2]
    # lns_shallow = df.index[df["elevation_condition"] == 3]
    ## region_center_ln = random.choice(list(lns_ocean) + list(lns_coast) + list(lns_land) + list(lns_shallow))
    ## region_center_ln = random.choice(df.index[df["elevation_condition"] != -1])
    region_center_ln = random.choice(lns_coast)
    region_center_pc = icm.get_point_code_from_prefix_lookup_number(region_center_ln)

    circle_radius_dist = lambda: power_law() * region_radius_gc

    # to choose existing point file (they are just lists of the point codes in a given area)
    pc_dir = "PointFiles"
    # fname, fp = icdb.get_random_point_code_file(pc_dir)
    # fname = "pcs_in_db_2022-11-22_K003201212211_0.02438586135795968.txt"  # small region
    # fname = "pcs_in_db_2022-11-22_D030022203113_0.11116449190061897.txt"  # region on coast of Oligra with a good amount of both land and sea
    # fname = "pcs_in_db_2022-11-22_K03302310332_0.1443687630207259.txt"
    # fp = os.path.join(pc_dir, fname)
    # db.verify_points_exist(pcs_with_data_in_region)
    ## region_center_pc, region_radius_gc = icdb.get_point_code_and_distance_from_filename(fname)

    ## region_center_pc, region_radius_gc = "D030022203113", 0.11116449190061897
    ## region_center_pc, region_radius_gc = "B", 0.01
    ## region_center_pc, region_radius_gc = "D3", 0.05  # all-land area of Oligra, for testing condition interpolation
    ## region_center_pc, region_radius_gc = "C1", 0.05
    ## region_center_pc, region_radius_gc = "A", 0.10
    ## region_center_pc, region_radius_gc = "C102", 0.046

    # turn interpolation on if you're doing a new region so it can get the elevation conditions at all the points you want
    # should probably always do this anyway to double check that we have conditions everywhere
    interpolate_condition_variables = True

    # pc_fp = icdb.get_point_code_filename(region_center_pc, region_radius_gc, prefix_str="pcs_in_db", parent_dir=pc_dir)
    # if True: #not os.path.exists(pc_fp):
    #     # should probably just redo this all the time anyway in case another run added some points
    #     icdb.make_point_code_file_for_region(db, region_center_pc, region_radius_gc)
    # pcs_with_data_in_region = icdb.get_point_codes_from_file(pc_fp)
    lns_with_data_in_region = icdb.get_lookup_numbers_in_database_in_region(db, region_center_pc, region_radius_gc, xyzg, use_narrowing=True, lns_to_consider=None)

    region_radius_gc_km = region_radius_gc * planet_radius_km
    center_latlondeg = icm.get_latlon_from_point_code(region_center_pc, xyzg)

    print(f"region centered at {region_center_pc} {center_latlondeg} deg with radius {region_radius_gc_km} km")
    # pu.scatter_icosa_points_by_code(pcs_with_data_in_region)

    pcs_in_region_at_resolution = icm.get_region_around_point_code_by_spreading(region_center_pc, region_radius_gc, xyzg, resolution_iterations=resolution_iterations)
    lns_in_region_at_resolution = icm.get_prefix_lookup_numbers_from_point_codes(pcs_in_region_at_resolution)

    # add missing points to the dataframe
    missing_lns = [x for x in lns_in_region_at_resolution if x not in df.index]
    missing_df = pd.DataFrame(index=missing_lns, columns=df.columns)
    condition_variables = db.get_condition_variables()
    value_variables = db.get_value_variables()
    missing_df.loc[:, condition_variables] = -1
    missing_df.loc[:, value_variables] = 0
    for var in condition_variables:
        missing_df[var] = missing_df[var].astype(icdb.IcosahedronPointDatabase.CONDITION_DTYPE)
    for var in value_variables:
        missing_df[var] = missing_df[var].astype(icdb.IcosahedronPointDatabase.VALUE_DTYPE)

    # debug dtypes
    # missing_df = missing_df.astype("int64")  # why is it float even when I declared dtype int and set the values as ints? I have no idea
    print(f"\nmissing_df:\n{missing_df}\n")
    print(f"\nmissing_df.dtypes:\n{missing_df.dtypes}\n")
    concat_df = pd.concat([df, missing_df]).sort_index()
    db.df = concat_df
    db.write_hdf()

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

    condition_shorthand_dict = lmd.get_condition_shorthand_dict(world_name="Cada II", map_variable="elevation")
    el_cond_to_default_value = lmd.get_default_values_of_conditions(world_name="Cada II", map_variable="elevation")
    el_cond_to_min_value = {sh: condition_shorthand_dict[sh]["min"] for sh in condition_shorthand_dict}
    el_cond_to_max_value = {sh: condition_shorthand_dict[sh]["max"] for sh in condition_shorthand_dict}

    if interpolate_condition_variables:
        try:
            lns_to_edit = interpolate_conditions(db, lns_with_data_in_region, lns_in_region_at_resolution, planet_radius_km, el_cond_to_default_value, xyzg)
        except InterpolationError:
            print("interpolation failed; stopping this region generation")
            # this way it doesn't stop the program completely, since this is a rare occurrence and we can just ignore it and try again with a different region
            return
    else:
        lns_to_edit = list(lns_in_region_at_resolution)
    
    # start by generating random elevation circles in the region
    print("getting xyzs and distance matrix")
    pcs_to_edit = icm.get_point_codes_from_prefix_lookup_numbers(lns_to_edit)
    xyz_array = icm.get_xyz_array_from_point_codes(pcs_to_edit, xyzg)
    xyz_tuples = [tuple(xyz) for xyz in xyz_array]
    xyz_dict = BiDict.from_dict(dict(zip(lns_to_edit, xyz_tuples)))
    matrix = GreatCircleDistanceMatrix(xyz_array, radius=1)
    print("-- done getting xyzs and distance matrix")
    
    el_cond_by_ln = db.get_series(lns_to_edit, "elevation_condition")
    el_by_ln = db.get_series(lns_to_edit, "elevation")
    min_el_by_ln = lmd.translate_array_by_dict(el_cond_by_ln, el_cond_to_min_value)
    max_el_by_ln = lmd.translate_array_by_dict(el_cond_by_ln, el_cond_to_max_value)
    rises = max_el_by_ln - el_by_ln
    falls = min_el_by_ln - el_by_ln
    assert (rises[~pd.isna(rises)] >= 0).all(), "some points currently are above their maximum elevation"
    assert (falls[~pd.isna(falls)] <= 0).all(), "some points currently are below their minimum elevation"

    el_df = pd.DataFrame(index=lns_to_edit, columns=["el_cond", "min_el", "el", "max_el"])
    el_df["el_cond"] = el_cond_by_ln
    el_df["min_el"] = min_el_by_ln
    el_df["el"] = el_by_ln
    el_df["max_el"] = max_el_by_ln
    el_df["min_d_el"] = falls
    el_df["max_d_el"] = rises
    print("working with these elevation conditions:")
    print(el_df)

    changes = pd.Series({ln: 0 for ln in lns_to_edit})  # keep track of what we changed in this region
    n_passed = 0
    n_failed = 0
    for c_i in range(n_circles):
        if c_i % 100 == 0:
            condition_pass_rate = n_passed / c_i if c_i > 0 else np.nan
            print(f"circle {c_i} / {n_circles}; condition pass rate so far = {condition_pass_rate}")
        circle_radius_gc = circle_radius_dist()
        ln_center = random.choice(lns_to_edit)
        xyz_center = np.array(xyz_dict[ln_center])
        p_xyzs_in_circle = matrix.get_points_within_distance_of_point(xyz_center, circle_radius_gc)
        lns_in_circle = [xyz_dict[p_xyz] for p_xyz in p_xyzs_in_circle.keys()]

        d_el = int(round(np.random.normal(0, el_stdev)))

        # keep track of how much each point can move up or down from where it is right now, 
        # and adjust d_el toward 0 (but keep its direction) 
        # so that the conditions are all still met 
        # (unless that change becomes zero in which case just start over)
        # just look at absolute value of changes that are in the same direction as d_el
        old_els_this_circle = el_df.loc[lns_in_circle, "el"]
        # Series.min()/max() won't be NaN unless the whole thing is NaN
        min_d_el_this_circle = el_df.loc[lns_in_circle, "min_d_el"].max()
        max_d_el_this_circle = el_df.loc[lns_in_circle, "max_d_el"].min()
        
        is_rise = d_el >= 0
        if is_rise:
            d_el = min(d_el, max_d_el_this_circle)
        else:
            d_el = max(d_el, min_d_el_this_circle)
        
        if abs(d_el) < 1:
            # d_el reached zero, conditions failed on this circle
            n_failed += 1
            continue
        else:
            n_passed += 1

        assert d_el % 1 == 0, f"{d_el} of type {type(d_el)}"
        new_els = old_els_this_circle + d_el
        # check new_els still meet elevation conditions
        min_els_this_circle = el_df.loc[lns_in_circle, "min_el"]
        max_els_this_circle = el_df.loc[lns_in_circle, "max_el"]
        has_min_mask = ~pd.isna(min_els_this_circle)
        has_max_mask = ~pd.isna(max_els_this_circle)
        falls_this_circle = min_els_this_circle[has_min_mask] - new_els[has_min_mask]
        rises_this_circle = max_els_this_circle[has_max_mask] - new_els[has_max_mask]
        assert (falls_this_circle <= 0).all(), "some minimum elevation is now violated"
        assert (rises_this_circle >= 0).all(), "some maximum elevation is now violated"

        # print("conditions passed, adding to db")
        db.add_value(lns_in_circle, "elevation", d_el)
        changes[lns_in_circle] += d_el
        # print(f"added {d_el}")
        # print("new values:", db[pcs_in_circle, "elevation"])

    assert n_passed + n_failed == n_circles
    print(f"condition pass rate overall = {n_passed / n_circles}")

    # print("plotting elevation after generating noise")
    # pu.plot_variables_scattered_from_db(db, points_to_edit, ["elevation_condition", "elevation"])
    pu.plot_variables_interpolated_from_db(db, lns_to_edit, ["elevation_condition", "elevation"], xyzg, resolution=1000, show=False)
    plt.gcf().set_size_inches(18, 6)
    now_str = datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
    plt.savefig(f"ElevationImages/GeneratedElevation_{region_center_pc}_{region_radius_gc}_{now_str}.png")
    plt.gcf().clear()

    # plot the changes so I can monitor what it's doing
    pu.plot_variable_interpolated_from_dict(changes, xyzg, resolution=1000, show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(f"ElevationImages/ElevationChanges_{region_center_pc}_{region_radius_gc}_{now_str}.png")
    plt.gcf().clear()

    if True: #input("write these results? y/n (default n)") == "y":
        db.write_hdf()



if __name__ == "__main__":
    db_root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"
    db = IcosahedronPointDatabase.load(db_root_dir)
    planet_radius_km = icm.CADA_II_RADIUS_KM
    xyzg = XyzLookupAncestryGraph()  # will add to it as needed

    ## region_center_latlondeg, region_radius_great_circle_km = (10, -87), 1000  # Western Amphoto
    ## region_center_latlondeg, region_radius_great_circle_km = (-87, 10), 1000  # somewhere in O-Z because I originally mixed up latlon
    ## region_center_latlondeg, region_radius_great_circle_km = (90, 0), 2000  # North Pole
    ## region_center_latlondeg, region_radius_great_circle_km = (-14, -115), 2000  # Thiuy-Rainia Bay
    ## region_center_latlondeg, region_radius_great_circle_km = (86.5, -13), 250  # small region in Tomar Strait in Mienta, for testing on smaller regions
    ## region_center_latlondeg, region_radius_great_circle_km = (25, -84), 2000  # Jhorju
    ## region_center_latlondeg, region_radius_great_circle_km = (-54.28119589256169, 175.64265081464623), 250  # random from 2022-07-16
    ## region_center_latlondeg, region_radius_great_circle_km = (26.083351229768834, 94.04570559120195), 2000  # northern Mienta, from a random point
    ## region_center_point_code = icm.get_nearest_icosa_point_to_latlon(region_center_latlondeg, maximum_distance=1, planet_radius=icm.CADA_II_RADIUS_KM)

    while True:
        run_region_generation(db, planet_radius_km, xyzg)
    