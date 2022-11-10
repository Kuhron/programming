from IcosahedronPointDatabase import IcosahedronPointDatabase
import IcosahedronMath as icm
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm
from GreatCircleDistanceMatrix import GreatCircleDistanceMatrix
from BiDict import BiDict
import LoadMapData
import PlottingUtil as pu

import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def filter_point_numbers_in_region_one_by_one(point_numbers, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    # goes over all point numbers you pass in, gets their distance from the center,
    # and returns the ones that are within the radius

    # gc = great-circle distance
    region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km
    point_numbers_in_region = set()
    t0 = time.time()
    n_points = len(point_numbers)
    for i, pn in enumerate(point_numbers):
        # pc = icm.get_point_code_from_point_number(pn)
        # print(pc)  # to see what kind of iteration precision we are dealing with here
        if i % 100 == 0 and i != 0:
            dt = time.time() - t0
            rate = i / dt
            time_remaining = (n_points - i) / rate
            print(f"i = {i} / {len(point_numbers)}; estimated {time_remaining:.2f} seconds remaining")
            print("TODO there has got to be a way to use the point codes to infer from ancestry that the point is too far away from the region, so I can filter them faster without actually calculating their position")
        latlondeg = icm.get_latlon_from_point_number(pn)
        d_gc = UnitSpherePoint.distance_great_circle_latlondeg_static(region_center_latlondeg, latlondeg)
        # print(f"distance from center {region_center_latlondeg}\nto point #{pn} {latlondeg}\nis {d_gc} normalized to sphere radius 1")
        if d_gc <= region_radius_gc_normalized:
            point_numbers_in_region.add(pn)
            print("added point", pn)
    point_numbers_in_region = sorted(point_numbers_in_region)
    return point_numbers_in_region


def filter_point_numbers_in_region_all_at_once(point_numbers, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    raise Exception("deprecated; one-by-one should be fast enough now")
    # # try doing like nearest neighbors or large-scale distance query or something
    # region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km
    # xyzs = icm.get_xyzs_from_point_numbers(point_numbers)
    # xyz = mcm.unit_vector_lat_lon_to_cartesian(region_center_latlondeg)
    # distances = UnitSpherePoint.distance_3d_xyzs_to_xyz_static(xyzs, xyz)
    # mask = distances <= region_radius_gc_normalized
    # return point_numbers[mask]


def get_point_number_cache_fp(region_center_latlondeg, region_radius_great_circle_km):
    return f"PointNumberCache_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km.txt"


def get_point_numbers_in_region_from_db(db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    point_numbers_in_db = db.get_all_point_numbers_with_data()
    print(f"checking {len(point_numbers_in_db)} points")
    
    # try different algorithms for finding the correct set of points
    filter_point_numbers_in_region = filter_point_numbers_in_region_one_by_one
    # filter_point_numbers_in_region = filter_point_numbers_in_region_all_at_once
    
    point_numbers_in_region_in_db = filter_point_numbers_in_region(point_numbers_in_db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km)
    
    # in case of crash
    print("---- point_numbers_in_region_in_db ----")
    print(point_numbers_in_region_in_db)
    print("//// point_numbers_in_region_in_db ////")

    write_point_numbers_to_cache(point_numbers_in_region_in_db, region_center_latlondeg, region_radius_great_circle_km)
    return point_numbers_in_region_in_db


def write_point_numbers_to_cache(point_numbers, region_center_latlondeg, region_radius_great_circle_km):
    point_number_cache_fp = get_point_number_cache_fp(region_center_latlondeg, region_radius_great_circle_km)
    # keep everything that's already there
    try:
        with open(point_number_cache_fp) as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    existing_pns = set(int(l.strip()) for l in lines)
    point_numbers = set(point_numbers)
    if len(point_numbers - existing_pns) == 0:
        print("all points are already cached")
        return

    point_numbers |= existing_pns
    with open(point_number_cache_fp, "w") as f:
        for pn in sorted(point_numbers):
            f.write(f"{pn}\n")


def read_point_numbers_from_cache(region_center_latlondeg, region_radius_great_circle_km):
    point_number_cache_fp = get_point_number_cache_fp(region_center_latlondeg, region_radius_great_circle_km)
    if os.path.exists(point_number_cache_fp):
        with open(point_number_cache_fp) as f:
            lines = f.readlines()
        point_numbers = [int(l.strip()) for l in lines]
        return point_numbers
    else:
        raise FileNotFoundError(point_number_cache_fp)


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


def get_points_in_region(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations):
    print("getting points in region")
    try:
        res = get_points_in_region_from_file(region_center_latlondeg, region_radius_great_circle_km, iterations)
        print("got points from file")
    except FileNotFoundError:
        print("calculating points in region using icosa math")
        points = get_points_in_region_raw(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations)
        fp = get_points_at_resolution_cache_fp(region_center_latlondeg, region_radius_great_circle_km, iterations)
        with open(fp, "w") as f:
            f.write("\n".join(str(pn) for pn in points))
        res = points
    print("-- done getting points in region")
    return res


def get_points_at_resolution_cache_fp(region_center_latlondeg, region_radius_great_circle_km, iterations):
    return f"PointsAtResolution_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km_iterations_{iterations}.txt"


def get_points_in_region_from_file(region_center_latlondeg, region_radius_great_circle_km, iterations):
    fp = get_points_at_resolution_cache_fp(region_center_latlondeg, region_radius_great_circle_km, iterations)
    with open(fp) as f:
        lines = f.readlines()
    return [int(l.strip()) for l in lines]


def get_points_in_region_raw(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations):
    # procedure: start with icosa starting points
    # have function which tells you the farthest a point's descendants can get from it
    # calculate the point's distance from the region center
    # if that distance - max_distance_of_descendant is still too far away, then throw this point out and don't bother looking at its descendants

    region_center_xyz = mcm.unit_vector_lat_lon_to_cartesian(*region_center_latlondeg, deg=True)
    distance = lambda pn: icm.get_distance_icosa_point_to_xyz_great_circle(pn, region_center_xyz, radius=planet_radius_km)
    points_in_region = []
    points_whose_children_could_be_in_region = []

    # for the poles, just check whether they're in the region or not, since they don't have descendants
    for pn in range(2):
        # print(f"checking point {pn}")
        d = distance(pn)
        # print(f"distance from {pn} to region center is {d}")
        if d < region_radius_great_circle_km:
            # print(f"{pn} is in the region")
            points_in_region.append(pn)

    starting_points = list(range(2, 12))
    to_check = starting_points
    for iteration in range(0, iterations+1):
        n_to_check = len(to_check)
        if n_to_check == 0:
            break
        to_check_next_round = []
        print(f"checking iteration {iteration}")
        # for each point, only check its actual distance on the first time you see it
        for i, pn in enumerate(to_check):
            if i % 100 == 0 and i != 0:
                print(f"{i}/{n_to_check} this round (iteration {iteration})")
            # print(f"checking point {pn}")
            if iteration == icm.get_iteration_born_from_point_number(pn):
                # check its distance, put it in points_in_region if it fits
                d = distance(pn)
                # print(f"distance from {pn} to region center is {d}")
                if d < region_radius_great_circle_km:
                    # print(f"{pn} is in the region")
                    points_in_region.append(pn)
            else:
                # print(f"already seen point {pn} by iteration {iteration} because it was born at iteration {icm.get_iteration_born(pn)}")
                pass

            # now, if its children *starting after this iteration* can ever get into the region, keep it in to_check and add the children as well, else throw it out
            
            iteration_of_next_child = iteration + 1
            should_check_descendants = descendants_of_point_can_ever_be_in_region(pn, region_center_xyz, region_radius_great_circle_km, planet_radius_km, iteration_of_next_child)
            if should_check_descendants:
                # print("should check descendants")
                to_check_next_round.append(pn)
                children = icm.get_children_from_point_number(pn, iteration+1)
                # print(f"children: {children}")
                to_check_next_round += children

        to_check = to_check_next_round

    return points_in_region


def descendants_of_point_can_ever_be_in_region(pn, region_center_xyz, region_radius_great_circle_km, planet_radius_km, iteration_of_next_child):
    # print(f"checking descendant min distance to region center, pn={pn}, iteration_of_next_child={iteration_of_next_child}")
    farthest_distance_descendant_can_be_from_pn = icm.get_farthest_distance_descendant_can_be(pn, radius=planet_radius_km, iteration_of_next_child=iteration_of_next_child)
    # print(f"farthest distance of descendant from pn: {farthest_distance_descendant_can_be_from_pn}")
    current_distance = icm.get_distance_icosa_point_to_xyz_great_circle(pn, region_center_xyz, radius=planet_radius_km)
    # print(f"current distance from pn to region center: {current_distance}")
    closest_descendant_can_be_to_region_center = max(0, current_distance - farthest_distance_descendant_can_be_from_pn)
    # print(f"closest descendant can be to region center: {closest_descendant_can_be_to_region_center}")
    return closest_descendant_can_be_to_region_center <= region_radius_great_circle_km


def interpolate_at_points_nearest_neighbor(points_to_interpolate_at, points_to_interpolate_from, variable_name, db, max_nn_distance=None):
    # interpolate the conditions at the points_at_this_resolution
    print("interpolating nearest neighbor")
    known_values = db[points_to_interpolate_from, variable_name]
    assert type(known_values) is dict
    if len(set(points_to_interpolate_at) - set(points_to_interpolate_from)) == 0:
        # already know values of all these points, don't bother getting xyz or doing nearest neighbor calculation
        return {pn: known_values[pn] for pn in points_to_interpolate_at}

    nn_pn_lookup, d_lookup = icm.get_nearest_neighbors_pn_to_pn_with_distance(query_pns=points_to_interpolate_at, candidate_pns=points_to_interpolate_from)
    interpolated = {}
    for i, pn in enumerate(points_to_interpolate_at):
        if i % 100 == 0:
            print(f"interpolating at points; progress {i}/{len(points_to_interpolate_at)}")
        if pn in known_values:
            # we already know its condition, no need to do nearest neighbors
            interpolated[pn] = known_values[pn]
        else:
            d = d_lookup[pn]
            nn_pn = nn_pn_lookup[pn]
            if max_nn_distance is None or d <= max_nn_distance:
                el_cond = db[nn_pn, variable_name]
                interpolated[pn] = el_cond
            else:
                # don't interpolate here, the nearest neighbor is too far away
                interpolated[pn] = None
    print("-- done interpolating nearest neighbor")
    return interpolated


def plot_variable_scattered(db, point_numbers, var_to_plot, show=True):
    print(f"plotting variable scattered: {var_to_plot}")
    pn_to_val = db[point_numbers, var_to_plot]
    # print(pn_to_val)
    latlons = [icm.get_latlon_from_point_number(pn) for pn in point_numbers]
    lats = [latlon[0] for latlon in latlons]
    lons = [latlon[1] for latlon in latlons]
    vals = [pn_to_val.get(pn) for pn in point_numbers]
    plt.scatter(lons, lats, c=vals)
    plt.colorbar()
    plt.title(var_to_plot)
    if show:
        plt.show()


def plot_variables_scattered(db, point_numbers, vars_to_plot):
    print("plotting variables scattered")
    n_plots = len(vars_to_plot)
    for i, var in enumerate(vars_to_plot):
        plt.subplot(1, n_plots, i+1)
        plot_variable_scattered(db, point_numbers, var, show=False)
    plt.show()


def plot_variable_interpolated(db, point_numbers, var_to_plot, resolution, show=True):
    print(f"plotting variable interpolated: {var_to_plot}")
    latlons = [icm.get_latlon_from_point_number(pn) for pn in point_numbers]
    values_dict = db[point_numbers, var_to_plot]
    # print(values_dict)
    values = [values_dict.get(pn) for pn in point_numbers]
    pu.plot_interpolated_data(latlons, values, lat_range=None, lon_range=None, n_lats=resolution, n_lons=resolution, with_axis=True)
    if show:
        plt.show()


def plot_variables_interpolated(db, point_numbers, vars_to_plot, resolution):
    print("plotting variables interpolated")
    for var in vars_to_plot:
        plot_variable_interpolated(db, point_numbers, var, resolution, show=False)
        plt.title(var)
        plt.show()
    # don't do subplots here because the PlottingUtil code sets its own fig/ax


def plot_latlons(point_numbers):
    latlons = [icm.get_latlon_from_point_number(pn) for pn in point_numbers]
    lats = [latlon[0] for latlon in latlons]
    lons = [latlon[1] for latlon in latlons]
    plt.scatter(lons, lats)
    plt.show()



if __name__ == "__main__":
    root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"
    db = IcosahedronPointDatabase.load(root_dir)
    print("loaded db")

    # region_center_latlondeg, region_radius_great_circle_km = (10, -87), 1000  # Western Amphoto
    # region_center_latlondeg, region_radius_great_circle_km = (-87, 10), 1000  # somewhere in O-Z because I originally mixed up latlon
    # region_center_latlondeg, region_radius_great_circle_km = (90, 0), 2000  # North Pole
    # region_center_latlondeg, region_radius_great_circle_km = (-14, -115), 2000  # Thiuy-Rainia Bay
    # region_center_latlondeg, region_radius_great_circle_km = (86.5, -13), 250  # small region in Tomar Strait in Mienta, for testing on smaller regions
    region_center_latlondeg, region_radius_great_circle_km = (25, -84), 2000  # Jhorju
    # region_center_latlondeg, region_radius_great_circle_km = (-54.28119589256169, 175.64265081464623), 250  # random from 2022-07-16
    # region_center_latlondeg, region_radius_great_circle_km = (26.083351229768834, 94.04570559120195), 2000  # northern Mienta, from a random point
    region_center_point_code = icm.get_nearest_icosa_point_to_latlon(region_center_latlondeg, maximum_distance=1, planet_radius=icm.CADA_II_RADIUS_KM)

    # to choose random one
    # region_center_point_code = icm.get_random_point_code(min_iterations=3, expected_iterations=6)
    # region_center_latlondeg = icm.get_latlon_from_point_code(region_center_point_code)
    # region_radius_great_circle_km = 2000
    # region_center_latlondeg, region_radius_great_circle_km = (
    #    UnitSpherePoint.get_random_unit_sphere_point().latlondeg(), 250
    # )
    print(f"region centered at {region_center_point_code} {region_center_latlondeg} deg with radius {region_radius_great_circle_km} km")

    planet_radius_km = icm.CADA_II_RADIUS_KM
    power_law_param = 0.25  # 1 is uniform dist, >1 is more weight toward 1 and less toward 0, a=0 is all weight at 0, a=inf is all weight at 1
    power_law = lambda: np.random.power(power_law_param)
    circle_radius_dist = lambda: power_law() * region_radius_great_circle_km
    el_stdev = 15
    n_circles = 10000

    # # just test how fast it is to get the positions of all points in the database
    # point_numbers_in_database = db.get_all_point_numbers_with_data()
    # for i, pn in enumerate(point_numbers_in_database):
    #     xyz = icm.get_xyz_from_point_number(pn)
    #     if i % 100 == 0:
    #         print(i, pn, xyz)

    point_numbers_in_db = db.get_all_point_numbers_with_data()
    points_with_data_in_region = get_point_numbers_with_data_in_region(db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km)

    # DEBUG
    # points_with_data_in_region = random.sample(points_with_data_in_region, 100)

    # use this to check if the point locations look right 
    # (is it actually interpolating conditions onto icosa lattice points, for instance? 
    # (which should locally look like a triangular/hexagonal lattice 
    # with random interloping image pixel points) 
    # or is it just taking the image pixels? 
    # (which should locally look like a rectangular lattice))
    # plot_variable_scattered(db, points_with_data_in_region, "elevation") 

    # plot_variable_scattered(db, points_with_data_in_region, "elevation_condition")
    # plot_variable_interpolated(db, points_with_data_in_region, "elevation", resolution=1000)
    # input("press enter to continue")

    # edit the region and then plot again

    interpolate = True  # only do this when you don't have points in file? but it should be able to know that all those points already have elevation condition (FIXME) and so leaving this as True shouldn't be an issue
    if interpolate:
        # raise Exception("FIXME! It will overwrite the existing data with default elevation values if you use interpolate=True")
        # interpolate condition at other points as nearest neighbor
        # (with some max distance to that neighbor so we don't get things like 
        # the middle of the ocean thinking it has to be a coast/shallow 
        # because that's what's on the edge of the nearest image thousands of km away)
        edge_length_of_resolution_km = 100
        iterations_of_resolution = icm.get_iterations_needed_for_edge_length(edge_length_of_resolution_km, planet_radius_km)
        print(f"resolution needs {iterations_of_resolution} iterations of icosa")
        print("TODO maybe cache this too (in a file like the point numbers, so we have one cache of point numbers with data and another of point numbers in certain region at certain resolution, although maybe only the latter is necessary and then you can easily check the database for which ones have the variable defined)")
        n_points_total_at_this_iteration = icm.get_n_points_from_iterations(iterations_of_resolution)
        # points_at_this_resolution_in_region = filter_point_numbers_in_region(list(range(n_points_total_at_this_iteration)), region_center_latlondeg, region_radius_great_circle_km, planet_radius_km)  # include points of previous iterations  # too long, brute force over the whole planet
        points_at_this_resolution_in_region = get_points_in_region(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations=iterations_of_resolution)
        print(f"{len(points_at_this_resolution_in_region)} points in region")
        # plot_latlons(points_at_this_resolution_in_region)

        # so using the points in the region with data as interpolation, we will generate elevations at the points_at_this_resolution AND the points that already have data
        points_to_interpolate_at = list(set(points_at_this_resolution_in_region) | set(points_with_data_in_region))

        interpolated_elevation_conditions = interpolate_at_points_nearest_neighbor(
            points_to_interpolate_at=points_to_interpolate_at,
            points_to_interpolate_from=points_with_data_in_region,
            variable_name="elevation_condition",
            db=db,
            max_nn_distance=100/planet_radius_km,
        )
        # write these to the db
        print(f"got interpolated elevation conditions, writing {len(interpolated_elevation_conditions)} items to db")
        # input("press enter to continue")
        point_numbers_to_cache = points_at_this_resolution_in_region
        for pn, el_cond in interpolated_elevation_conditions.items():
            if el_cond is None:
                # interpolation failed because neighbors were too far away
                continue
            old_el_cond = db[pn, "elevation_condition"]
            if old_el_cond is not None and el_cond != old_el_cond:
                raise RuntimeError(f"elevation condition changed: {old_el_cond} -> {el_cond}")
            if old_el_cond is None:
                db[pn, "elevation_condition"] = el_cond
            point_numbers_to_cache.append(pn)
            print(f"p #{pn} had old elevation condition {old_el_cond}, new {el_cond}")
        if input("write these results? y/n (default n)") == "y":
            db.write()
        write_point_numbers_to_cache(point_numbers_to_cache, region_center_latlondeg, region_radius_great_circle_km)
        points_to_edit = list(set(interpolated_elevation_conditions.keys()) | set(points_with_data_in_region))  # want both the new points and the points already having data
    else:
        print("not interpolating, just using points that already have db data")
        points_to_edit = points_with_data_in_region

    # start by generating random elevation circles in the region
    xyz_array = icm.get_xyz_array_from_point_numbers(points_to_edit)
    xyz_tuples = [tuple(xyz) for xyz in xyz_array]
    xyz_dict = BiDict.from_dict(dict(zip(points_to_edit, xyz_tuples)))
    matrix = GreatCircleDistanceMatrix(xyz_array, radius=planet_radius_km)

    # db.add_variable("elevation")
    shorthand_dict = LoadMapData.get_condition_shorthand_dict(world_name="Cada II", map_variable="elevation")
    elevation_conditions = db[points_to_edit, "elevation_condition"]
    elevation_condition_to_default_value = LoadMapData.get_default_values_of_conditions(world_name="Cada II", map_variable="elevation")
    elevation_condition_to_min_value = {sh: shorthand_dict[sh]["min"] for sh in shorthand_dict}
    elevation_condition_to_max_value = {sh: shorthand_dict[sh]["max"] for sh in shorthand_dict}

    # set_unknown_

    # set elevations to default value for condition
    for pn in points_to_edit:
        el = db[pn, "elevation"]
        if el is None:
            # print(f"got None for elevation at pn={pn}")
            # input("check")
            el_cond = elevation_conditions[pn]
            val = elevation_condition_to_default_value[el_cond]
            db[pn, "elevation"] = int(round(val))
        else:
            # print(f"got {el} for elevation at pn={pn}")
            pass

    n_passed = 0
    n_failed = 0
    for c_i in range(n_circles):
        print(f"circle {c_i} / {n_circles}")
        circle_radius_gc = circle_radius_dist()
        pn_center = random.choice(points_to_edit)
        xyz_center = np.array(xyz_dict[pn_center])
        # distances = matrix.get_distances_to_point(xyz_center)
        p_xyzs_in_circle = matrix.get_points_within_distance_of_point(xyz_center, circle_radius_gc)
        pns_in_circle = [xyz_dict[p_xyz] for p_xyz in p_xyzs_in_circle.keys()]
        # print(f"this circle contains {pns_in_circle}")

        d_el = int(round(np.random.normal(0, el_stdev)))
        old_els = db[pns_in_circle, "elevation"]
        # print(f"values in old elevations: {sorted(set(old_els.values()))}")  # debugging when it is overwriting existing data with default elevations

        # keep track of how much each point can move up or down from where it is right now, and adjust d_el toward 0 (but keep its direction) so that the conditions are all still met (unless that change becomes zero in which case just start over)
        # just look at absolute value of changes that are in the same direction as d_el
        is_rise = d_el >= 0
        d_el = abs(d_el)
        max_move = abs(d_el)
        for pn in pns_in_circle:
            el = old_els[pn]
            # print(f"point {pn} has old_el {el}")
            el_cond = elevation_conditions[pn]
            if is_rise:
                # check if we would go over the max
                max_val = elevation_condition_to_max_value[el_cond]
                if max_val is not None:
                    assert max_val >= el, f"invalid elevation found: {el} exceeds max value of {max_val} at point {pn}"
                    assert type(max_val) is int, el_cond
                    move_size = abs(max_val - el)
                    d_el = min(d_el, move_size)
            else:
                min_val = elevation_condition_to_min_value[el_cond]
                if min_val is not None:
                    assert el >= min_val, f"invalid elevation found: {el} is below min value of {min_val} at point {pn}"
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
        assert type(d_el) is int

        new_els = {pn: old_els[pn] + d_el for pn in pns_in_circle}
        # check new_els still meet elevation conditions
        all_meet_conditions = True
        for pn in pns_in_circle:
            el_cond = elevation_conditions[pn]
            max_val = elevation_condition_to_max_value[el_cond]
            min_val = elevation_condition_to_min_value[el_cond]
            new_val = new_els[pn]
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
            d_els = [d_el] * len(pns_in_circle)
            db.add_values(pns_in_circle, "elevation", d_els)
            # print(f"added {d_el}")
            # print("new values:", db[pns_in_circle, "elevation"])
        else:
            print("conditions failed, making new circle")
            raise Exception("this shouldn't happen anymore")

    assert n_passed + n_failed == n_circles
    print(f"condition pass rate {n_passed / n_circles}")

    plot_variable_interpolated(db, points_to_edit, "elevation", resolution=1000)
    if input("write these results? y/n (default n)") == "y":
        db.write()

