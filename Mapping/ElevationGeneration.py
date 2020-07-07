import random
import time
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


if __name__ == "__main__":
    from_image = False
    from_data = True
    generate_further_elevation_changes = True
    
    image_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/"
    if from_image:
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
        lattice_edge_length_km = 500
        print("creating map lattice")
        map_lattice = IcosahedralGeodesicLattice(edge_length_km=lattice_edge_length_km)
        print("- done creating map lattice")
        print("creating ElevationGenerationMap from image")
        m = ElevationGenerationMap.from_image(image_fp, color_condition_dict, default_color, latlon00, latlon01, latlon10, latlon11, map_lattice)
        print("- done creating ElevationGenerationMap")
        m.freeze_coastlines()
        generate_initial_elevation_changes = True
    elif from_data:
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_CircleIsland.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_CircleIsland50x50.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_LegronCombinedDigitization_ThinnedBorders_Final.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_MientaDigitization_ThinnedBorders_Final.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_MientaDigitization_ThinnedBorders_Final_FurtherChanges.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_OligraZitomoDigitization_ThinnedBorders_Final.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_OligraZitomoDigitization_ThinnedBorders_Final_FurtherChanges.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Mako.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Amphoto.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Jhorju.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Ilausa.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Ilausa_FurtherChanges.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Ilausa_FurtherChanges_Bay.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_NorthernMystIslands.txt"
        # data_fp_no_dir = "TestElevationData10x10.txt"
        # if using Cada WorldMapScanPNGs:
        # data_fp = image_dir + data_fp_no_dir

        # data_fp = "/home/wesley/programming/Mapping/ElevationGenerationOutputData_Random-20200707-004152.txt"
        # data_fp = "/home/wesley/programming/Mapping/ElevationGenerationOutputData_Random-20200707-004152_FurtherChanges-20200707-004639.txt"
        data_fp = "/home/wesley/programming/Mapping/ElevationGenerationOutputData_Random-20200707-004152_FurtherChanges-20200707-004639_FurtherChanges-20200707-005113.txt"

        print("from data {}".format(data_fp))
        # latlon00, latlon01, latlon10, latlon11 = [(25, -15), (20, 10), (-2, -8), (2, 12)]
        latlon00, latlon01, latlon10, latlon11 = None, None, None, None
        m = ElevationGenerationMap.load_elevation_data(data_fp, latlon00, latlon01, latlon10, latlon11)

        generate_initial_elevation_changes = False
        if generate_further_elevation_changes:
            elevation_data_output_fp = data_fp.replace(".txt", "_FurtherChanges.txt")
            plot_image_output_fp = data_fp.replace("OutputData", "OutputPlot").replace(".txt", "_FurtherChanges.png")
    else:
        lattice = IcosahedralGeodesicLattice(edge_length_km=250)
        m = ElevationGenerationMap(lattice)

        m.fill_all(0)
        elevation_data_output_fp = "/home/wesley/programming/Mapping/ElevationGenerationOutputData_Random.txt"
        plot_image_output_fp = "/home/wesley/programming/Mapping/ElevationGenerationOutputPlot_Random.png"
        generate_initial_elevation_changes = True
        
    print("map size {} pixels".format(m.size()))

    if generate_initial_elevation_changes:
        print("generating initial elevation changes")
        expected_change_size = 100
        expected_touches_per_point = 25
        n_steps = int(expected_touches_per_point / expected_change_size * m.size())
        # n_steps = np.inf
        # n_steps = 10000
        plot_every_n_steps = None
        print("filling elevation for {} steps, plotting every {}".format(n_steps, plot_every_n_steps))
        m.fill_elevations(n_steps, expected_change_size, plot_every_n_steps)
        m.plot()
        if input("save data? (y/n, default n)\n").strip().lower() == "y":
            m.save_elevation_data(elevation_data_output_fp)
        if input("save image? (y/n, default n)\n").strip().lower() == "y":
            m.save_plot_image(plot_image_output_fp, size_inches=(36, 24))
        print("- done generating initial elevation changes")
    elif generate_further_elevation_changes:
        print("generating further elevation changes")
        m.unfreeze_all()  # allow coastlines to change
        expected_change_size = 50
        expected_touches_per_point = 200
        n_steps = int(expected_touches_per_point / expected_change_size * m.size())
        plot_every_n_steps = None
        print("making further elevation changes for {} steps, plotting every {}".format(n_steps, plot_every_n_steps))
        m.fill_elevations(n_steps, expected_change_size, plot_every_n_steps)
        if input("save data? (y/n, default n)\n").strip().lower() == "y":
            m.save_elevation_data(elevation_data_output_fp)
        if input("save image? (y/n, default n)\n").strip().lower() == "y":
            m.save_plot_image(plot_image_output_fp, size_inches=(36, 24))
        print("- done generating further elevation changes")
    else:
        m.plot()
        # m.plot_map_and_gradient_magnitude()
        # m.create_flow_arrays()
        # m.plot_flow_amounts()
        # m.plot_rivers()
        # m.plot_flow_steps(10000)
        # m.plot_average_water_location()
        print("- done plotting")
