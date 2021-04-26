from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
import IcosahedronMath
import NoiseMath as nm
import PlottingUtil as pu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def test_generate_whole_planet():
    edge_length_km = None
    iterations = 6
    test_lattice = IcosahedralGeodesicLattice(edge_length_km=edge_length_km, iterations=iterations)

    # making example images and data for each type of noise generation function
    df = test_lattice.create_dataframe(with_coords=True)
    # df = nm.add_random_data_circles(df, "elevation_expectation", n_patches=1000)
    # df["elevation_expectation_omega"] = np.repeat(0.5, len(df.index))
    df = nm.add_random_data_circles(df, "elevation", n_patches=1000)
    # df = nm.add_random_data_radial_waves(df, "elevation", n_waves=1000, expected_amplitude=100)
    # df = nm.add_random_data_jagged_patches(df, "elevation", test_lattice.adjacencies, test_lattice.get_index_of_usp, n_patches=1000)
    # df = nm.add_random_data_spikes(df, "elevation", n_spikes=len(df.index), sigma=100)
    # df = nm.add_random_data_independent_all_points(df, "elevation", n_iterations=1000, sigma=10)
    # h_stretch_parameters = np.random.uniform(0, 1, 200)**0.25  # skew towards 1
    # df = nm.add_random_data_sigmoid_decay_hills(df, "elevation", n_hills=200, h_stretch_parameters=h_stretch_parameters)  # pointier hills
    # df = nm.add_random_data_sigmoid_decay_hills(df, "elevation", n_hills=200, h_stretch_parameters=np.repeat(0, 200))  # wider plateaus
    # test_lattice.plot_data(df, "elevation_expectation", equirectangular=True, save=True, size_inches=(48, 24))
    test_lattice.plot_data(df, "elevation", equirectangular=True, save=False, size_inches=(48, 24), contour_lines=True)

    plt.show()


def test_generate_on_section_of_condition_data():
    min_lat, max_lat = -15, 15
    min_lon, max_lon = -30, 30

    condition_iterations = 3
    icosa_usps_with_conditions = IcosahedronMath.get_usps_in_latlon_rectangle(min_lat, max_lat, min_lon, max_lon, condition_iterations, IcosahedronMath.STARTING_POINTS)
    condition_latlons = [p.latlondeg() for p in icosa_usps_with_conditions]
    condition_lats = [ll[0] for ll in condition_latlons]
    condition_lons = [ll[1] for ll in condition_latlons]

    data_iterations = 4
    icosa_usps_with_data = IcosahedronMath.get_usps_in_latlon_rectangle(min_lat, max_lat, min_lon, max_lon, data_iterations, IcosahedronMath.STARTING_POINTS)

    elevation_conditions = {}
    color_by_condition = {
        "sea": (0,1,1,1), 
        # "coast": (0,38/255,1,1), 
        "land": (1,0.5,0,1), 
        # "shallow": (0,148/255,1,1)
    }
    condition_colors_lst = []
    for pi, p in enumerate(icosa_usps_with_conditions):
        condition = random.choice(list(color_by_condition.keys()))
        elevation_conditions[pi] = condition
        condition_colors_lst.append(color_by_condition[condition])

    elevation_condition_ranges = {
        "sea": (None, -100),
        "land": (100, None),
        "coast": (-15, 15),
        "shallow": (-5, 0),
    }
    data_points = icosa_usps_with_data  # can try doing even more points inside this, e.g. get conditions for only 6 iterations but generate data on 7

    # noise generation subject to the condition functions
    df = pd.DataFrame(index=[p.point_number for p in data_points])
    df["elevation"] = [0 for p in data_points]
    df["xyz"] = [p.xyz() for p in data_points]
    elevation_conditions_by_point = [elevation_conditions.get(p.point_number) for p in data_points]
    elevation_ranges_by_point = [elevation_condition_ranges.get(condition) for condition in elevation_conditions_by_point]

    df["min_elevation"] = pd.Series(data=[r[0] if r is not None else None for r in elevation_ranges_by_point], index=df.index)
    df["max_elevation"] = pd.Series(data=[r[1] if r is not None else None for r in elevation_ranges_by_point], index=df.index)

    n_patches = 100
    df = nm.add_random_data_circles(df, "elevation", n_patches=n_patches)
    elevations = {p: df.loc[p.point_number, "elevation"] for p in data_points}

    # now get the data and interpolate to plot
    data_coords = [p.latlondeg() for p in data_points]
    values = [elevations[p] for p in data_points]
    lat_range = [min_lat, max_lat]
    lon_range = [min_lon, max_lon]
    n_lats = 100
    n_lons = 200
    pu.plot_interpolated_data(data_coords, values, lat_range, lon_range, n_lats, n_lons, with_axis=True)
    plt.scatter(condition_lons, condition_lats, facecolors="none", edgecolors=condition_colors_lst)  # facecolors "none" and edgecolors defined is how you make open circle markers (so it's easier to see what value is at that point)
    plt.show()


if __name__ == "__main__":
    # test_generate_whole_planet()
    test_generate_on_section_of_condition_data()
    print("done")
