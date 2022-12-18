import IcosahedronMath as icm
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm

import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt



def filter_point_codes_in_region_one_by_one(pcs, center_pc, radius_gc_normalized):
    # goes over all point numbers you pass in, gets their distance from the center,
    # and returns the ones that are within the radius

    print(f"filtering {len(pcs)} points for inclusion in region")
    region = set()
    t0 = time.time()
    last_print_time = time.time()
    n_points = len(pcs)
    region_center_latlondeg = icm.get_latlon_from_point_code(center_pc)
    for i, pc in enumerate(pcs):
        if time.time() - last_print_time > 2:
            dt = time.time() - t0
            rate = i / dt
            time_remaining = (n_points - i) / rate
            print(f"i = {i} / {len(pcs)}; estimated {time_remaining:.2f} seconds remaining")
            last_print_time = time.time()
        latlondeg = icm.get_latlon_from_point_code(pc)
        d_gc = UnitSpherePoint.distance_great_circle_latlondeg_static(region_center_latlondeg, latlondeg)
        # print(f"distance from center {region_center_latlondeg}\nto point {pc} {latlondeg}\nis {d_gc} normalized to sphere radius 1")
        if d_gc <= radius_gc_normalized:
            region.add(pc)
            # print("added point", pc)
    region = sorted(region)
    print(f"done filtering {len(pcs)} points for inclusion in region")
    return region


def filter_point_numbers_in_region_all_at_once(point_numbers, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    raise Exception("deprecated; one-by-one should be fast enough now")
    # # try doing like nearest neighbors or large-scale distance query or something
    # region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km
    # xyzs = icm.get_xyzs_from_point_numbers(point_numbers)
    # xyz = mcm.unit_vector_lat_lon_to_cartesian(region_center_latlondeg)
    # distances = UnitSpherePoint.distance_3d_xyzs_to_xyz_static(xyzs, xyz)
    # mask = distances <= region_radius_gc_normalized
    # return point_numbers[mask]
