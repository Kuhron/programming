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
import numpy as np
import matplotlib.pyplot as plt


def filter_point_numbers_in_region(point_numbers, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    # gc = great-circle distance
    region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km
    point_numbers_in_region = set()
    for i, pn in enumerate(point_numbers):
        if i % 1000 == 0:
            print(f"i = {i} / {len(point_numbers)}")
        latlondeg = icm.get_latlon_from_point_number(pn)
        d_gc = UnitSpherePoint.distance_great_circle_latlondeg_static(region_center_latlondeg, latlondeg)
        if d_gc <= region_radius_gc_normalized:
            point_numbers_in_region.add(pn)
            print("added point", pn)
    point_numbers_in_region = sorted(point_numbers_in_region)
    return point_numbers_in_region


def get_point_number_cache_fp(region_center_latlondeg, region_radius_great_circle_km):
    return f"PointNumberCache_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km.txt"


def get_point_numbers_in_region_from_db(db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    point_numbers_in_db = db.get_all_point_numbers_with_data()
    print(f"checking {len(point_numbers_in_db)} points")
    region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km
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
    try:
        return read_point_numbers_from_cache(region_center_latlondeg, region_radius_great_circle_km)
    except FileNotFoundError:
        region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km
        point_numbers = list(get_point_numbers_in_region_from_db(db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km))
        return point_numbers


def get_points_in_region(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations):
    try:
        return get_points_in_region_from_file(region_center_latlondeg, region_radius_great_circle_km, iterations)
    except FileNotFoundError:
        points = get_points_in_region_raw(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations)
        fp = get_points_at_resolution_cache_fp(region_center_latlondeg, region_radius_great_circle_km, iterations)
        with open(fp, "w") as f:
            f.write("\n".join(str(pn) for pn in points))
        return points


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
        if len(to_check) == 0:
            break
        to_check_next_round = []
        print(f"checking iteration {iteration}")
        # for each point, only check its actual distance on the first time you see it
        for pn in to_check:
            # print(f"checking point {pn}")
            if iteration == icm.get_iteration_born(pn):
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
                children = icm.get_children(pn, iteration+1)
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
    known_values = db[points_to_interpolate_from, variable_name]
    assert type(known_values) is dict
    if len(set(points_to_interpolate_at) - set(points_to_interpolate_from)) == 0:
        # already know values of all these points, don't bother getting xyz or doing nearest neighbor calculation
        return {pn: known_values[pn] for pn in points_to_interpolate_at}

    interpolated = {}
    xyzs_with_data = {icm.get_xyz_from_point_number(pn): pn for pn in points_to_interpolate_from}
    for pn in points_to_interpolate_at:
        if pn in known_values:
            # we already know its condition, no need to do nearest neighbors
            interpolated[pn] = known_values[pn]
        else:
            xyz = icm.get_xyz_from_point_number(pn)
            nn_xyz, d = icm.get_nearest_neighbor_xyz_to_xyz(xyz, list(xyzs_with_data.keys()))
            if max_nn_distance is None or d <= max_nn_distance:
                nn_pn = xyzs_with_data[nn_xyz]
                el_cond = db[nn_pn, variable_name]
                interpolated[pn] = el_cond
            else:
                # don't interpolate here, the nearest neighbor is too far away
                interpolated[pn] = None
    return interpolated


def plot_variable_scattered(db, point_numbers, var_to_plot, show=True):
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
    n_plots = len(vars_to_plot)
    for i, var in enumerate(vars_to_plot):
        plt.subplot(1, n_plots, i+1)
        plot_variable_scattered(db, point_numbers, var, show=False)
    plt.show()


def plot_variable_interpolated(db, point_numbers, var_to_plot, resolution, show=True):
    latlons = [icm.get_latlon_from_point_number(pn) for pn in point_numbers]
    values_dict = db[point_numbers, var_to_plot]
    # print(values_dict)
    values = [values_dict.get(pn) for pn in point_numbers]
    pu.plot_interpolated_data(latlons, values, lat_range=None, lon_range=None, n_lats=resolution, n_lons=resolution, with_axis=True)
    if show:
        plt.show()


def plot_variables_interpolated(db, point_numbers, vars_to_plot, resolution):
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

    # region_center_latlondeg = (10, -87)  # Western Amphoto
    # region_center_latlondeg = (-87, 10)  # somewhere in O-Z because I originally mixed up latlon
    # region_center_latlondeg = (90, 0)  # North Pole
    region_center_latlondeg, region_radius_great_circle_km = (-14, -115), 2000  # Thiuy-Rainia Bay
    region_radius_great_circle_km = 2000
    planet_radius_km = icm.CADA_II_RADIUS_KM

    points_with_data_in_region = get_point_numbers_with_data_in_region(db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km)

    # DEBUG
    # points_with_data_in_region = random.sample(points_with_data_in_region, 100)

    # plot_variable_scattered(db, points_with_data_in_region, "elevation_condition")
    # plot_variable_interpolated(db, points_with_data_in_region, "elevation", resolution=1000)
    # input("press enter to continue")

    # edit the region and then plot again
    # interpolate condition at other points as nearest neighbor (with some max distance to that neighbor so we don't get things like the middle of the ocean thinking it has to be a coast/shallow because that's what's on the edge of the nearest image thousands of km away)
    edge_length_of_resolution_km = 10
    iterations_of_resolution = icm.get_iterations_needed_for_edge_length(edge_length_of_resolution_km, planet_radius_km)
    print(f"resolution needs {iterations_of_resolution} iterations of icosa")
    print("TODO maybe cache this too (in a file like the point numbers, so we have one cache of point numbers with data and another of point numbers in certain region at certain resolution, although maybe only the latter is necessary and then you can easily check the database for which ones have the variable defined)")
    n_points_total_at_this_iteration = icm.get_points_from_iterations(iterations_of_resolution)
    # points_at_this_resolution_in_region = filter_point_numbers_in_region(list(range(n_points_total_at_this_iteration)), region_center_latlondeg, region_radius_great_circle_km, planet_radius_km)  # include points of previous iterations  # too long, brute force over the whole planet
    points_at_this_resolution_in_region = get_points_in_region(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations=iterations_of_resolution)
    print(f"{len(points_at_this_resolution_in_region)} points in region")
    # plot_latlons(points_at_this_resolution_in_region)

    # so using the points in the region with data as interpolation, we will generate elevations at the points_at_this_resolution AND the points that already have data

    interpolate = True
    if interpolate:
        interpolated_elevation_conditions = interpolate_at_points_nearest_neighbor(points_to_interpolate_at=points_at_this_resolution_in_region, points_to_interpolate_from=points_with_data_in_region, variable_name="elevation_condition", db=db, max_nn_distance=100/planet_radius_km)
        # write these to the db
        print(f"got interpolated elevation conditions, writing {len(interpolated_elevation_conditions)} items to db")
        input("press enter to continue")
        point_numbers_to_cache = points_at_this_resolution_in_region
        for pn, el_cond in interpolated_elevation_conditions.items():
            if el_cond is None:
                # interpolation failed because neighbors were too far away
                continue
            db[pn, "elevation_condition"] = el_cond
            point_numbers_to_cache.append(pn)
        db.write()
        write_point_numbers_to_cache(point_numbers_to_cache, region_center_latlondeg, region_radius_great_circle_km)
        points_to_edit = list(interpolated_elevation_conditions.keys())
    else:
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

    # set elevations to default value for condition
    for pn in points_to_edit:
        el_cond = elevation_conditions[pn]
        val = elevation_condition_to_default_value[el_cond]
        if db[pn, "elevation"] is None:
            db[pn, "elevation"] = int(round(val))

    power_law_param = 0.25
    power_law = lambda: np.random.power(power_law_param)
    circle_radius_dist = lambda: power_law() * region_radius_great_circle_km
    n_circles = 1000
    for c_i in range(n_circles):
        print(f"circle {c_i} / {n_circles}")
        circle_radius_gc = circle_radius_dist()
        pn_center = random.choice(points_to_edit)
        xyz_center = np.array(xyz_dict[pn_center])
        # distances = matrix.get_distances_to_point(xyz_center)
        p_xyzs_in_circle = matrix.get_points_within_distance_of_point(xyz_center, circle_radius_gc)
        pns_in_circle = [xyz_dict[p_xyz] for p_xyz in p_xyzs_in_circle.keys()]
        # print(f"this circle contains {pns_in_circle}")

        d_el = int(round(np.random.normal(0, 10)))
        old_els = db[pns_in_circle, "elevation"]
        new_els = {pn: old_els[pn] + d_el for pn in pns_in_circle}
        # check new_els still meet elevation conditions
        all_meet_conditions = True
        for pn in pns_in_circle:
            el_cond = elevation_conditions[pn]
            min_val = elevation_condition_to_min_value[el_cond]
            max_val = elevation_condition_to_max_value[el_cond]
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

    plot_variable_interpolated(db, points_to_edit, "elevation", resolution=1000)
    if input("write these results? y/n (default n)") == "y":
        db.write()

