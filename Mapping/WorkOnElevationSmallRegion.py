from IcosahedronPointDatabase import IcosahedronPointDatabase
import IcosahedronMath as icm
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm

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
            print("i =", i)
        latlondeg = icm.get_latlon_from_point_number(pn)
        d_gc = UnitSpherePoint.distance_great_circle_latlondeg_static(region_center_latlondeg, latlondeg)
        if d_gc <= region_radius_gc_normalized:
            point_numbers_in_region.add(pn)
            print("added point", pn)
    point_numbers_in_region = sorted(point_numbers_in_region)
    return point_numbers_in_region


def get_point_numbers_in_region_from_db(db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    point_numbers_in_db = db.get_all_point_numbers_with_data()
    print(f"checking {len(point_numbers_in_db)} points")
    region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km
    point_numbers_in_region_in_db = filter_point_numbers_in_region(point_numbers_in_db, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km)

    point_number_cache_fp = f"PointNumberCache_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km.txt"
    with open(point_number_cache_fp, "w") as f:
        for pn in point_numbers_in_region_in_db:
            f.write(f"{pn}\n")
    return point_numbers_in_region_in_db


def get_point_numbers_with_data_in_region(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    point_number_cache_fp = f"PointNumberCache_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km.txt"
    if os.path.exists(point_number_cache_fp):
        with open(point_number_cache_fp) as f:
            lines = f.readlines()
        point_numbers = [int(l.strip()) for l in lines]
    else:
        region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km
        point_numbers = list(get_point_numbers_in_region_from_db(db, region_center_latlondeg, region_radius_gc_normalized))
    return point_numbers


def get_points_in_region(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations):
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


def plot_variable(db, point_numbers, var_to_plot):
    vals = [db[pn, var_to_plot] for pn in point_numbers]
    latlons = [icm.get_latlon_from_point_number(pn) for pn in point_numbers]
    lats = [latlon[0] for latlon in latlons]
    lons = [latlon[1] for latlon in latlons]
    plt.scatter(lons, lats, c=vals)
    plt.show()


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

    region_center_latlondeg = (10, -87)  # Western Amphoto
    # region_center_latlondeg = (-87, 10)  # somewhere in O-Z because I originally mixed up latlon
    # region_center_latlondeg = (90, 0)  # North Pole
    # region_center_latlondeg = (-14, -115)  # Thiuy-Rainia Bay
    region_radius_great_circle_km = 1000
    planet_radius_km = icm.CADA_II_RADIUS_KM

    point_numbers = get_point_numbers_with_data_in_region(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km)
    # plot_variable(db, point_numbers, "elevation_condition")

    # edit the region and then plot again
    # interpolate condition at other points as nearest neighbor (with some max distance to that neighbor so we don't get things like the middle of the ocean thinking it has to be a coast/shallow because that's what's on the edge of the nearest image thousands of km away)
    edge_length_of_resolution_km = 10
    iterations_of_resolution = icm.get_iterations_needed_for_edge_length(edge_length_of_resolution_km, planet_radius_km)
    print(f"resolution needs {iterations_of_resolution} iterations of icosa")
    n_points_total_at_this_iteration = icm.get_points_from_iterations(iterations_of_resolution)
    # points_at_this_resolution_in_region = filter_point_numbers_in_region(list(range(n_points_total_at_this_iteration)), region_center_latlondeg, region_radius_great_circle_km, planet_radius_km)  # include points of previous iterations  # too long, brute force over the whole planet
    points_at_this_resolution_in_region = get_points_in_region(region_center_latlondeg, region_radius_great_circle_km, planet_radius_km, iterations=iterations_of_resolution)
    print("points in region:", points_at_this_resolution_in_region)
    plot_latlons(points_at_this_resolution_in_region)
