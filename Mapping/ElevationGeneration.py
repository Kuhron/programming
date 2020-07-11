import random
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from datetime import datetime, timedelta

import MapCoordinateMath as mcm
from ElevationGenerationMap import ElevationGenerationMap
from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
from LatitudeLongitudeLattice import LatitudeLongitudeLattice


def confirm_overwrite_file(output_fp):
    if os.path.exists(output_fp):
        yn = input("Warning! Overwriting file {}\ncontinue? (y/n, default n)".format(output_fp))
        if yn != "y":
            print("aborting")
            return False
    return True


def get_expected_change_size_from_user(n_points_total):
    inp = input("expected change size as proportion of sphere surface area (if float in (0, 1)) or number of points (if int >= 1): ")
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


def get_expected_touches_per_point_from_user():
    return int(input("expected touches per point (suggestions: 10-100): "))




if __name__ == "__main__":
    from_image = input("from image? (y/n) ").strip().lower() == "y"
    if from_image:
        from_data = False
        generate_initial_elevation_changes = True
        generate_further_elevation_changes = False
    else:
        from_data = input("from data? (y/n) ").strip().lower() == "y"
        if from_data:
            generate_initial_elevation_changes = False
            generate_further_elevation_changes = input("generate further changes? (y/n) ").strip().lower() == "y"
        else:
            print("generating new data at random")
            generate_initial_elevation_changes = True
            generate_further_elevation_changes = False
    
    image_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/"
    if from_image:

        # DANGER OF MEMORY LEAKS if use big maps! Watch top!
        # image_fp_no_dir = "LegronCombinedDigitization_ThinnedBorders_Final.png"
        # image_fp_no_dir = "MientaDigitization_ThinnedBorders_Final.png"
        # image_fp_no_dir = "OligraZitomoDigitization_ThinnedBorders_Final.png"

        # image_fp_no_dir = "TestMap3_ThinnedBorders.png"
        # image_fp_no_dir = "TestMap_NorthernMystIslands.png"
        # image_fp_no_dir = "TestMap_Jhorju.png"
        # image_fp_no_dir = "TestMap_Amphoto.png"
        # image_fp_no_dir = "TestMap_Mako.png"
        # image_fp_no_dir = "TestMap_Myst.png"
        # image_fp_no_dir = "TestMap_Ilausa.png"
        # image_fp_no_dir = "TestMap_VerticalStripes.png"
        # image_fp_no_dir = "TestMap_AllLand.png"
        # image_fp_no_dir = "TestMap_CircleIsland.png"
        # image_fp_no_dir = "TestMap_CircleIsland50x50.png"
        image_fp = image_dir + image_fp_no_dir

        print("from image {}".format(image_fp))

        elevation_data_output_fp = image_dir + "EGD_" + image_fp_no_dir.replace(".png", ".txt")
        plot_image_output_fp = image_dir + "EGP_" + image_fp_no_dir
    
        color_condition_dict = {
            # (  0,  38, 255, 255): (0,  lambda x: x == 0, True),  # dark blue = sea level
            (  0, 255, 255, 255): (-1, lambda x: x < 0, False),  # cyan = sea
            (255, 255, 255, 255): (1, lambda x: x > 0, False),  # white = land
            (  0,   0,   0, 255): (0, lambda x: True, False),  # black = unspecified, anything goes
            # (  0, 255,  33, 255): (1,  lambda x: x > 0 or defect(), False),  # green = land
            # (255,   0,   0, 255): (1,  lambda x: x > 0 or defect(), False),  # red = land (country borders)
        }
        default_color = (0, 0, 0, 255)
        latlon00, latlon01, latlon10, latlon11 = [(30, -30), (30, 30), (-30, -30), (-30, 30)]
        print("creating map lattice")
        map_lattice = IcosahedralGeodesicLattice(iterations=6)
        print("- done creating map lattice")
        print("creating ElevationGenerationMap from image")
        m = ElevationGenerationMap.from_image(image_fp, color_condition_dict, default_color, latlon00, latlon01, latlon10, latlon11, map_lattice)
        print("- done creating ElevationGenerationMap")
        m.freeze_coastlines()
    elif from_data:
        # data_fp_no_dir = "EGD_TestMap_CircleIsland.txt"
        # data_fp_no_dir = "EGD_TestMap_CircleIsland50x50.txt"
        # data_fp_no_dir = "EGD_LegronCombinedDigitization_ThinnedBorders_Final.txt"
        # data_fp_no_dir = "EGD_MientaDigitization_ThinnedBorders_Final.txt"
        # data_fp_no_dir = "EGD_MientaDigitization_ThinnedBorders_Final_FurtherChanges.txt"
        # data_fp_no_dir = "EGD_OligraZitomoDigitization_ThinnedBorders_Final.txt"
        # data_fp_no_dir = "EGD_OligraZitomoDigitization_ThinnedBorders_Final_FurtherChanges.txt"
        # data_fp_no_dir = "EGD_TestMap_Mako.txt"
        # data_fp_no_dir = "EGD_TestMap_Amphoto.txt"
        # data_fp_no_dir = "EGD_TestMap_Jhorju.txt"
        # data_fp_no_dir = "EGD_TestMap_Ilausa.txt"
        # data_fp_no_dir = "EGD_TestMap_Ilausa_FurtherChanges.txt"
        # data_fp_no_dir = "EGD_TestMap_Ilausa_FurtherChanges_Bay.txt"
        # data_fp_no_dir = "EGD_TestMap_NorthernMystIslands.txt"
        # data_fp_no_dir = "TestElevationData10x10.txt"
        # if using Cada WorldMapScanPNGs:
        # data_fp = image_dir + data_fp_no_dir

        project_name = input("project name to load: ")
        project_version = input("project version number to load: ")
        project_version_array = [int(x) for x in project_version.split("-")]
        project_dir = "/home/wesley/programming/Mapping/Projects/{}/".format(project_name)
        data_fp = project_dir + "Data/EGD_{0}_v{1}.txt".format(project_name, project_version)

        print("from data {}".format(data_fp))
        # latlon00, latlon01, latlon10, latlon11 = [(25, -15), (20, 10), (-2, -8), (2, 12)]
        latlon00, latlon01, latlon10, latlon11 = None, None, None, None
        m = ElevationGenerationMap.load_elevation_data(data_fp, latlon00, latlon01, latlon10, latlon11)

        generate_initial_elevation_changes = False
        if generate_further_elevation_changes:
            new_version = project_version_array[:-1] + [project_version_array[-1] + 1]
            new_version_number = "-".join(str(x) for x in new_version)
            print("loaded version {}, outputting version {}".format(project_version, new_version_number))
            # elevation_data_output_fp = project_dir + "Data/EGD_{0}_v{1}.txt".format(project_name, new_version_number)
            # plot_image_output_fp = project_dir + "Plots/EGP_{0}_v{1}.png".format(project_name, new_version_number)
            version_number = new_version_number
        else:
            # in case want to overwrite existing plot, e.g. after fixing plotting bugs
            # plot_image_output_fp = project_dir + "Plots/EGP_{0}_v{1}.png".format(project_name, project_version)
            version_number = project_version
    else:
        lattice = IcosahedralGeodesicLattice(iterations=6)
        m = ElevationGenerationMap(lattice)
        m.fill_all("elevation", 0)
        project_name = input("name for new project: ")
        project_dir = "/home/wesley/programming/Mapping/Projects/{}/".format(project_name)
        os.mkdir(project_dir)
        os.mkdir(project_dir + "Data/")
        os.mkdir(project_dir + "Plots/")
        version_number = 0
        # elevation_data_output_fp = project_dir + "Data/EGD_{0}_v{1}.txt".format(project_name, new_version_number)
        # plot_image_output_fp = project_dir + "Plots/EGP_{0}_v{1}.png".format(project_name, new_version_number)
        generate_initial_elevation_changes = True
        
    print("map size {} pixels".format(m.size()))

    if generate_initial_elevation_changes or generate_further_elevation_changes:
        if generate_initial_elevation_changes:
            print("generating initial elevation changes")
        else:
            print("generating further elevation changes")
            m.unfreeze_all()  # allow coastlines to change

        m.add_fault_lines(24)
        # m.add_hotspots(200)
        m.lattice.plot_data(m.data_dict, "volcanism")
        plt.show()

        n_points_total = m.size()
        expected_change_sphere_proportion = get_expected_change_size_from_user(n_points_total)
        expected_touches_per_point = get_expected_touches_per_point_from_user()
        n_steps = int(round(expected_touches_per_point / expected_change_sphere_proportion))
        plot_every_n_steps = None

        if generate_initial_elevation_changes:
            print("filling elevation for {} steps, plotting every {}".format(n_steps, plot_every_n_steps))
        else:
            print("making further elevation changes for {} steps, plotting every {}".format(n_steps, plot_every_n_steps))

        elevation_change_parameters = ElevationGenerationMap.get_elevation_change_parameters_from_config_file()
        m.fill_elevations(n_steps, expected_change_sphere_proportion, plot_every_n_steps, elevation_change_parameters=elevation_change_parameters)
        if True: #input("save data? (y/n, default n)\n").strip().lower() == "y":
            m.save_data("elevation", project_name, version_number)
        if True: #input("save image? (y/n, default n)\n").strip().lower() == "y":
            m.save_plot_image("elevation", project_name, version_number, size_inches=(36, 24))
            m.save_plot_image("volcanism", project_name, version_number, size_inches=(72, 48))

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
        if True: #input("save image? (y/n, default n)\n").strip().lower() == "y":
            m.save_plot_image("elevation", project_name, version_number, size_inches=(36, 24))
            m.save_plot_image("volcanism", project_name, version_number, size_inches=(36, 24))
        print("- done plotting")
