from IcosahedronPointDatabase import IcosahedronPointDatabase
import IcosahedronMath as icm
from UnitSpherePoint import UnitSpherePoint

import random
import os
import numpy as np
import matplotlib.pyplot as plt


def get_point_numbers_in_region_from_db(db, region_center_latlondeg, region_radius_gc_normalized):
    point_numbers_in_db = db.get_all_point_numbers_with_data()
    print(f"checking {len(point_numbers_in_db)} points")
    point_numbers_in_region_in_db = set()
    for i, pn in enumerate(point_numbers_in_db):
        if i % 1000 == 0:
            print("i =", i)
        latlondeg = icm.get_latlon_from_point_number(pn)
        d_gc = UnitSpherePoint.distance_great_circle_latlondeg_static(region_center_latlondeg, latlondeg)
        if d_gc <= region_radius_gc_normalized:
            point_numbers_in_region_in_db.add(pn)
            print("added point", pn)
    point_numbers_in_region_in_db = sorted(point_numbers_in_region_in_db)

    point_number_cache_fp = f"PointNumberCache_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km.txt"
    with open(point_number_cache_fp, "w") as f:
        for pn in point_numbers_in_region_in_db:
            f.write(f"{pn}\n")
    return point_numbers_in_region_in_db



if __name__ == "__main__":
    root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"
    db = IcosahedronPointDatabase.load(root_dir)
    print("loaded db")

    region_center_latlondeg = (-87, 10)  # Western Amphoto
    region_radius_great_circle_km = 1000
    planet_radius_km = icm.CADA_II_RADIUS_KM
    region_radius_gc_normalized = region_radius_great_circle_km / planet_radius_km

    point_number_cache_fp = f"PointNumberCache_center_{region_center_latlondeg[0]}_{region_center_latlondeg[1]}_radius_{region_radius_great_circle_km}km.txt"
    if os.path.exists(point_number_cache_fp):
        with open(point_number_cache_fp) as f:
            lines = f.readlines()
        point_numbers = [int(l.strip()) for l in lines]
    else:
        point_numbers = get_point_numbers_in_region_from_db(db, region_center_latlondeg, region_radius_gc_normalized)

    print(point_numbers)

    # TODO plot this region with whatever variable you're interested in
    # edit the region and then plot again

