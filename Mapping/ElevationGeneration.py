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


def get_land_and_sea_colormap():
    # see PrettyPlot.py
    linspace_cmap_forward = np.linspace(0, 1, 128)
    linspace_cmap_backward = np.linspace(1, 0, 128)
    blue_to_black = mcolors.LinearSegmentedColormap.from_list('BlBk', [
        mcolors.CSS4_COLORS["blue"], 
        mcolors.CSS4_COLORS["black"],
    ])
    land_colormap = mcolors.LinearSegmentedColormap.from_list('land', [
        mcolors.CSS4_COLORS["darkgreen"],
        mcolors.CSS4_COLORS["limegreen"],
        mcolors.CSS4_COLORS["gold"],
        mcolors.CSS4_COLORS["darkorange"],
        mcolors.CSS4_COLORS["red"],
        mcolors.CSS4_COLORS["saddlebrown"],
        mcolors.CSS4_COLORS["gray"],
        mcolors.CSS4_COLORS["white"],
        # mcolors.CSS4_COLORS[""],
    ])
    # colors_land = plt.cm.YlOrBr(linspace_cmap_backward)  # example of how to call existing colormap object
    colors_land = land_colormap(linspace_cmap_forward)
    colors_sea = blue_to_black(linspace_cmap_backward)
    colors = np.vstack((colors_sea, colors_land))
    colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return colormap


def confirm_overwrite_file(output_fp):
    if os.path.exists(output_fp):
        yn = input("Warning! Overwriting file {}\ncontinue? (y/n, default n)".format(output_fp))
        if yn != "y":
            print("aborting")
            return False
    return True


if __name__ == "__main__":
    lattice = IcosahedralGeodesicLattice(edge_length_km=1000)
    m = ElevationGenerationMap(lattice)

    from_image = True
    from_data = False
    generate_further_elevation_changes = False
    
    image_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/"
    if from_image:
        # image_fp_no_dir = "LegronCombinedDigitization_ThinnedBorders_Final.png"
        # image_fp_no_dir = "MientaDigitization_ThinnedBorders_Final.png"
        # image_fp_no_dir = "OligraZitomoDigitization_ThinnedBorders_Final.png"
        # image_fp_no_dir = "TestMap3_ThinnedBorders.png"
        # image_fp_no_dir = "TestMap_NorthernMystIslands.png"
        # image_fp_no_dir = "TestMap_Jhorju.png"
        # image_fp_no_dir = "TestMap_Amphoto.png"
        image_fp_no_dir = "TestMap_Mako.png"
        # image_fp_no_dir = "TestMap_Myst.png"
        # image_fp_no_dir = "TestMap_Ilausa.png"
        # image_fp_no_dir = "TestMap_VerticalStripes.png"
        # image_fp_no_dir = "TestMap_AllLand.png"
        # image_fp_no_dir = "TestMap_CircleIsland.png"
        # image_fp_no_dir = "TestMap_CircleIsland50x50.png"
        image_fp = image_dir + image_fp_no_dir

        print("from image {}".format(image_fp))

        elevation_data_output_fp = image_dir + "ElevationGenerationOutputData_" + image_fp_no_dir.replace(".png", ".txt")
        plot_image_output_fp = image_dir + "ElevationGenerationOutputPlot_" + image_fp_no_dir
    
        color_condition_dict = {
            # (  0,  38, 255, 255): (0,  lambda x: x == 0, True),  # dark blue = sea level
            (  0, 255, 255, 255): (-1, lambda x: x < 0, False),  # cyan = sea
            (  0,   0,   0, 255): (1, lambda x: x > 0, False),
            # (  0, 255,  33, 255): (1,  lambda x: x > 0 or defect(), False),  # green = land
            # (255,   0,   0, 255): (1,  lambda x: x > 0 or defect(), False),  # red = land (country borders)
        }
        default_color = (0, 0, 0, 255)
        m = ElevationGenerationMap.from_image(image_fp, color_condition_dict, default_color)
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
        data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Mako.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Amphoto.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Jhorju.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Ilausa.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Ilausa_FurtherChanges.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Ilausa_FurtherChanges_Bay.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_NorthernMystIslands.txt"
        # data_fp_no_dir = "TestElevationData10x10.txt"
        data_fp = image_dir + data_fp_no_dir
        print("from data {}".format(data_fp))
        latlon00, latlon01, latlon10, latlon11 = [(25, -15), (20, 10), (-2, -8), (2, 12)]
        m = Map.load_elevation_data(data_fp, latlon00, latlon01, latlon10, latlon11)
        
        # test geodesic
        print("testing geodesic")
        edge_length_km = 5000
        geod = m.get_geodesic_latlon_meshgrid(edge_length_km)
        print("done testing geodesic")

        generate_initial_elevation_changes = False
        if generate_further_elevation_changes:
            elevation_data_output_fp = data_fp.replace(".txt", "_FurtherChanges.txt")
            plot_image_output_fp = data_fp.replace("OutputData", "OutputPlot").replace(".txt", "_FurtherChanges.png")
    else:
        m = Map(300, 500)
        m.fill_all(0)
        elevation_data_output_fp = "/home/wesley/programming/ElevationGenerationOutputData_Random.png"
        plot_image_output_fp = "/home/wesley/programming/ElevationGenerationOutputPlot_Random.png"
        generate_initial_elevation_changes = True
        
    print("map size {} pixels".format(m.size()))

    if generate_initial_elevation_changes:
        expected_change_size = 10000
        expected_touches_per_point = 200
        n_steps = int(expected_touches_per_point / expected_change_size * m.size())
        # n_steps = np.inf
        # n_steps = 10000
        plot_every_n_steps = None
        print("filling elevation for {} steps, plotting every {}".format(n_steps, plot_every_n_steps))
        m.fill_elevations(n_steps, expected_change_size, plot_every_n_steps)
        # m.plot()
        m.save_elevation_data(elevation_data_output_fp)
        m.save_plot_image(plot_image_output_fp)
    elif generate_further_elevation_changes:
        m.unfreeze_all()  # allow coastlines to change
        expected_change_size = 10000
        expected_touches_per_point = 5
        n_steps = int(expected_touches_per_point / expected_change_size * m.size())
        plot_every_n_steps = None
        print("making further elevation changes for {} steps, plotting every {}".format(n_steps, plot_every_n_steps))
        m.fill_elevations(n_steps, expected_change_size, plot_every_n_steps)
        m.save_elevation_data(elevation_data_output_fp)
        m.save_plot_image(plot_image_output_fp)
    else:
        m.plot(projection="ortho")
        # m.plot_map_and_gradient_magnitude()
        # m.create_flow_arrays()
        # m.plot_flow_amounts()
        # m.plot_rivers()
        # m.plot_flow_steps(10000)
        # m.plot_average_water_location()
