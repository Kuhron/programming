import IcosahedronMath as icm
from UnitSpherePoint import UnitSpherePoint
import MapCoordinateMath as mcm

import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt



def filter_point_codes_in_region_one_by_one(pcs, center_pc, radius_gc_normalized, xyzg):
    # goes over all point numbers you pass in, gets their distance from the center,
    # and returns the ones that are within the radius
    mask = get_mask_points_in_region(pcs, center_pc, radius_gc_normalized, xyzg)
    return sorted(pc for pc, b in zip(pcs, mask) if b)


def get_mask_points_in_region(lns, center_ln, radius_gc_normalized, xyzg):
    print(f"filtering {len(lns)} points for inclusion in region")
    t0 = time.time()
    last_print_time = time.time()
    n_points = len(lns)
    mask = [None for i in range(n_points)]
    region_center_xyz = xyzg[center_ln]
    for i, ln in enumerate(lns):
        if time.time() - last_print_time > 2:
            dt = time.time() - t0
            rate = i / dt
            time_remaining = (n_points - i) / rate
            print(f"filtering point codes in region: {i} / {len(lns)}; estimated {time_remaining:.2f} seconds remaining")
            last_print_time = time.time()
        xyz = xyzg[ln]
        d_gc = UnitSpherePoint.distance_great_circle_xyz_static(region_center_xyz, xyz)
        # print(f"distance from center {region_center_latlondeg}\nto point {pc} {latlondeg}\nis {d_gc} normalized to sphere radius 1")
        in_region = d_gc <= radius_gc_normalized
        mask[i] = in_region
    mask = np.array(mask)
    print(f"done filtering {len(lns)} points for inclusion in region")
    return mask


def filter_point_numbers_in_region_all_at_once(point_numbers, region_center_latlondeg, region_radius_great_circle_km, planet_radius_km):
    raise Exception("deprecated; one-by-one should be fast enough now")
    # # try doing like nearest neighbors or large-scale distance query or something
    ##region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km
    # xyzs = icm.get_xyzs_from_point_numbers(point_numbers)
    # xyz = mcm.unit_vector_lat_lon_to_cartesian(region_center_latlondeg)
    # distances = UnitSpherePoint.distance_3d_xyzs_to_xyz_static(xyzs, xyz)
    # mask = distances <= region_radius_gc_normalized
    # return point_numbers[mask]
