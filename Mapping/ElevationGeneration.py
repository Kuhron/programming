import random
import time
import os
import json
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
# from mpl_toolkits.basemap import Basemap  # don't use Basemap anymore; IcosahedralGeodesicLattice can now plot data itself
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from datetime import datetime, timedelta

import MapCoordinateMath as mcm
import PlottingUtil as pu
from ElevationGenerationMap import ElevationGenerationMap
from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
from LatitudeLongitudeLattice import LatitudeLongitudeLattice



def get_parameter_input(var_name, default_value):
    inp = input("set param {} (or just press enter for default value of {}): ".format(var_name, default_value))
    try:
        return float(inp)
    except ValueError:
        return None


def get_parameters_from_config_file():
    fp = "ParamConfig.json"
    with open(fp) as f:
        d = json.load(f)
    return d


def save_config_for_version(param_dict, projects_dir, project_name, project_version):
    fp = os.path.join(projects_dir, "{project_name}/Data/ParamConfig_{project_name}_v{project_version}.json".format(**locals()))
    with open(fp, "w") as f:
        json.dump(param_dict, f)


def confirm_overwrite_file(output_fp):
    if os.path.exists(output_fp):
        yn = input("Warning! Overwriting file {}\ncontinue? (y/n, default n)".format(output_fp))
        if yn != "y":
            print("aborting")
            return False
    return True


def convert_expected_change_size_to_proportion(expected_change_size, n_points_total):
    inp = expected_change_size
    fl = float(inp)
    if 0 < fl < 1:
        # proportion; return it directly
        print("user supplied proportion: {}".format(fl))
        proportion = fl
    elif 1 <= fl:
        i = int(round(fl))
        print("user supplied number of points: {}".format(i))
        assert i <= n_points_total, "cannot change more than the total number of points, which is {}".format(n_points_total)
        proportion = i/n_points_total
    else:
        raise ValueError("invalid value for expected size: {}".format(fl))
    return proportion


def get_project_versions_in_data_dir(data_dir, project_name):
    regex = "EGD_{}_.*_v(.*).txt".format(project_name)
    versions = [re.match(regex, f).group(1) for f in os.listdir(data_dir) if re.match(regex, f)]
    return versions


def get_key_strs_in_data_dir(data_dir, project_name, project_version):
    regex = "EGD_{}_(.*)_v{}.txt".format(project_name, project_version)
    key_strs = [re.match(regex, f).group(1) for f in os.listdir(data_dir) if re.match(regex, f)]
    return key_strs


def get_map_and_version_from_image(projects_dir, project_name, image_names, image_latlons, color_conditions, condition_ranges):
    # cada_image_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/"

    # DANGER OF MEMORY LEAKS if use big maps! Watch top!
    # image_fp_no_dir = "TestMap_Mako.png"
    project_dir = os.path.join(projects_dir, project_name)
    image_dir = os.path.join(project_dir, "ImageImporting")
    image_filename_regex = "EGII_" + project_name + "_(?P<variable>\w+)_(?P<image_name>\w+).png"
    # EGII means Elevation Generation Input Image
    files_in_dir = os.listdir(image_dir)
    print(image_dir, files_in_dir)
    re_matches = [re.match(image_filename_regex, filename) for filename in files_in_dir]
    print(re_matches)
    raise

    print("from image {}".format(image_fp))

    elevation_data_output_fp = os.path.join(image_dir, "EGD_" + image_fp_no_dir.replace(".png", ".txt"))
    plot_image_output_fp = os.path.join(image_dir, "EGP_" + image_fp_no_dir)

    # color_condition_dict = {
    #     # (  0,  38, 255, 255): (0,  lambda x: x == 0, True),  # dark blue = sea level
    #     (  0, 255, 255, 255): (-1, lambda x: x < 0, False),  # cyan = sea
    #     (255, 255, 255, 255): (1, lambda x: x > 0, False),  # white = land
    #     (  0,   0,   0, 255): (0, lambda x: True, False),  # black = unspecified, anything goes
    #     # (  0, 255,  33, 255): (1,  lambda x: x > 0 or defect(), False),  # green = land
    #     # (255,   0,   0, 255): (1,  lambda x: x > 0 or defect(), False),  # red = land (country borders)
    # }
    default_color = (0, 0, 0, 255)
    latlon00, latlon01, latlon10, latlon11 = [(30, -30), (30, 30), (-30, -30), (-30, 30)]
    print("creating map lattice")
    map_lattice = IcosahedralGeodesicLattice(iterations=6)
    print("- done creating map lattice")
    print("creating ElevationGenerationMap from image")
    m = ElevationGenerationMap.from_images(image_fps, image_latlons, color_conditions, condition_ranges, map_lattice)
    print("- done creating ElevationGenerationMap")
    m.freeze_coastlines()
    new_project_version = 0
    return m, new_project_version


def get_map_and_version_from_data(projects_dir, project_name, load_project_version):
    project_dir = os.path.join(projects_dir, "{}/".format(project_name))
    data_dir = os.path.join(project_dir, "Data/")
    if load_project_version == -1:
        # use most recent version
        existing_versions = get_project_versions_in_data_dir(data_dir, project_name)
        existing_versions_int = [int(x) for x in existing_versions]  # only handle int versions for now
        load_project_version = sorted(existing_versions_int)[-1]
    project_version_array = [int(x) for x in str(load_project_version).split("-")]

    # latlon00, latlon01, latlon10, latlon11 = [(25, -15), (20, 10), (-2, -8), (2, 12)]
    key_strs = get_key_strs_in_data_dir(data_dir, project_name, load_project_version)
    print("found data files for keys {}".format(key_strs))
    m = ElevationGenerationMap.from_data(key_strs, project_name, load_project_version)

    if generate_further_elevation_changes:
        new_version_array = project_version_array[:-1] + [project_version_array[-1] + 1]
        new_project_version = "-".join(str(x) for x in new_version_array)
        print("loaded version {}, outputting version {}".format(load_project_version, new_project_version))
        # elevation_data_output_fp = project_dir + "Data/EGD_{0}_v{1}.txt".format(project_name, new_project_version) # use os.path.join
        # plot_image_output_fp = project_dir + "Plots/EGP_{0}_v{1}.png".format(project_name, new_project_version) # use os.path.join
    else:
        # in case want to overwrite existing plot, e.g. after fixing plotting bugs
        # plot_image_output_fp = project_dir + "Plots/EGP_{0}_v{1}.png".format(project_name, new_project_version) # use os.path.join
        new_project_version = load_project_version

    return m, new_project_version


def get_map_and_version_new(projects_dir):
    lattice = IcosahedralGeodesicLattice(iterations=6)
    m = ElevationGenerationMap(lattice)
    m.fill_all("elevation", 0)
    project_dir = os.path.join(projects_dir, "{}/".format(project_name))
    os.mkdir(project_dir)
    os.mkdir(os.path.join(project_dir, "Data/"))
    os.mkdir(os.path.join(project_dir, "Plots/"))
    new_project_version = 0

    return m, new_project_version
 

if __name__ == "__main__":
    params = get_parameters_from_config_file()

    big_abs = params["big_abs"]
    color_conditions = params["color_conditions"]
    condition_ranges = params["condition_ranges"]
    critical_abs = params["critical_abs"]
    expected_change_size_proportion_or_n_points = params["expected_change_size_proportion_or_n_points"]
    expected_touches_per_point = params["expected_touches_per_point"]
    from_data = params["from_data"]
    from_image = params["from_image"]
    generate_elevation_changes = params["generate_elevation_changes"]
    hotspot_max_magnitude_factor = params["hotspot_max_magnitude_factor"]
    hotspot_min_magnitude_factor = params["hotspot_min_magnitude_factor"]
    image_latlons = params["image_latlons"]
    image_names = params["image_names"]
    land_proportion = params["land_proportion"]
    load_project_version = params["load_project_version"]
    max_volcanism_change_magnitude = params["max_volcanism_change_magnitude"]
    max_volcanism_wavenumber = params["max_volcanism_wavenumber"]
    min_volcanism_wavenumber = params["min_volcanism_wavenumber"]
    mu_when_big = params["mu_when_big"]
    mu_when_critical = params["mu_when_critical"]
    mu_when_small = params["mu_when_small"]
    n_fault_tripoints = params["n_fault_tripoints"]
    n_hotspots = params["n_hotspots"]
    n_volcanism_steps = params["n_volcanism_steps"]
    plot_every_n_steps = params["plot_every_n_steps"]
    positive_feedback_in_elevation = params["positive_feedback_in_elevation"]
    project_name = params["project_name"]
    projects_dir = params["projects_dir"]
    reference_area_ratio_at_big_abs = params["reference_area_ratio_at_big_abs"]
    reference_area_ratio_at_sea_level = params["reference_area_ratio_at_sea_level"]
    sigma_when_big = params["sigma_when_big"]
    sigma_when_critical = params["sigma_when_critical"]
    sigma_when_small = params["sigma_when_small"]
    spikiness = params["spikiness"]
    volcanism_coefficient_for_elevation = params["volcanism_coefficient_for_elevation"]
    volcanism_exponent_for_elevation = params["volcanism_exponent_for_elevation"]

    # xxx = params["xxx"]
    # can also make these lines by running MakeParamCode.py

    if from_image:
        assert not from_data, "cannot import from both image and data"
        print("importing from image")
        generate_initial_elevation_changes = generate_elevation_changes
        generate_further_elevation_changes = False
        m, new_project_version = get_map_and_version_from_image(projects_dir, project_name, image_names, image_latlons, color_conditions, condition_ranges)
    elif from_data:
        print("importing from data")
        generate_initial_elevation_changes = False
        generate_further_elevation_changes = generate_elevation_changes
        m, new_project_version = get_map_and_version_from_data(projects_dir, project_name, load_project_version)
    else:
        if not(generate_elevation_changes):
            print("you selected neither importation nor generation; nothing will happen")
            sys.exit()
        print("generating new data at random")
        generate_initial_elevation_changes = generate_elevation_changes
        generate_further_elevation_changes = False
        m, new_project_version = get_map_and_version_new(projects_dir)
    
    save_config_for_version(params, projects_dir, project_name, new_project_version)
    n_points_total = m.size()
    print("map size {} pixels".format(n_points_total))
    expected_change_sphere_proportion = convert_expected_change_size_to_proportion(expected_change_size_proportion_or_n_points, n_points_total)

    if generate_initial_elevation_changes or generate_further_elevation_changes:
        if generate_initial_elevation_changes:
            print("generating initial elevation changes")
        else:
            print("generating further elevation changes")
            m.unfreeze_all()  # allow coastlines to change

        if generate_initial_elevation_changes:
            m.add_fault_lines(
                n_fault_tripoints=n_fault_tripoints,
                n_volcanism_steps=n_volcanism_steps,
                max_volcanism_change_magnitude=max_volcanism_change_magnitude,
                min_volcanism_wavenumber=min_volcanism_wavenumber,
                max_volcanism_wavenumber=max_volcanism_wavenumber,
            )
            m.add_hotspots(
                n_hotspots=n_hotspots,
                hotspot_min_magnitude_factor=hotspot_min_magnitude_factor,
                hotspot_max_magnitude_factor=hotspot_max_magnitude_factor,
            )

        n_points_total = m.size()
        n_steps = int(round(expected_touches_per_point / expected_change_sphere_proportion))

        if generate_initial_elevation_changes:
            print("filling elevation for {} steps, plotting every {}".format(n_steps, plot_every_n_steps))
        else:
            print("making further elevation changes for {} steps, plotting every {}".format(n_steps, plot_every_n_steps))

        if generate_further_elevation_changes:
            el_array = m.get_value_array("elevation")
            if max(el_array) - min(el_array) < 1:
                raise Exception("elevation array might be all zero; double check it was loaded properly")
        m.fill_elevations(
            n_steps=n_steps,
            plot_every_n_steps=plot_every_n_steps,
            expected_change_sphere_proportion=expected_change_sphere_proportion,
            positive_feedback_in_elevation=positive_feedback_in_elevation,
            reference_area_ratio_at_sea_level=reference_area_ratio_at_sea_level,
            reference_area_ratio_at_big_abs=reference_area_ratio_at_big_abs,
            big_abs=big_abs,
            critical_abs=critical_abs,
            mu_when_small=mu_when_small,
            mu_when_critical=mu_when_critical,
            mu_when_big=mu_when_big,
            sigma_when_small=sigma_when_small,
            sigma_when_critical=sigma_when_critical,
            sigma_when_big=sigma_when_big,
            land_proportion=land_proportion,
            spikiness=spikiness,
            volcanism_coefficient_for_elevation=volcanism_coefficient_for_elevation,
            volcanism_exponent_for_elevation=volcanism_exponent_for_elevation,
        )
        m.save_data("elevation", project_name, new_project_version)
        m.save_data("volcanism", project_name, new_project_version)
        m.save_plot_image("elevation", project_name, new_project_version, size_inches=(36, 24))
        if generate_initial_elevation_changes:
            m.save_plot_image("volcanism", project_name, new_project_version, size_inches=(72, 48), cmap=pu.get_volcanism_colormap())
        else:
            print("not saving volcanism plot because assumed it didn't change from version 0")  # so future self can notice why it's not saving if I decide to change this

        if generate_initial_elevation_changes:
            print("- done generating initial elevation changes")
        else:
            print("- done generating further elevation changes")

    else:
        # m.plot()
        # m.plot_map_and_gradient_magnitude()
        # m.create_flow_arrays()
        # m.plot_flow_amounts()
        # m.plot_rivers()
        # m.plot_flow_steps(10000)
        # m.plot_average_water_location()
        m.save_plot_image("elevation", project_name, new_project_version, size_inches=(36, 24))
        m.save_plot_image("volcanism", project_name, new_project_version, size_inches=(36, 24))
        print("- done plotting")
