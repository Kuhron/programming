from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
import IcosahedronMath
import NoiseMath as nm
import PlottingUtil as pu
import matplotlib.pyplot as plt
import numpy as np
import random


def test_generate_whole_planet():
    edge_length_km = None
    iterations = 4
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
    min_lat, max_lat = -5, 5
    min_lon, max_lon = -10, 10

    condition_iterations = 5
    icosa_usps_with_conditions = IcosahedronMath.get_usps_in_latlon_rectangle(min_lat, max_lat, min_lon, max_lon, condition_iterations, IcosahedronMath.STARTING_POINTS)

    data_iterations = 6
    icosa_usps_with_data = IcosahedronMath.get_usps_in_latlon_rectangle(min_lat, max_lat, min_lon, max_lon, data_iterations, IcosahedronMath.STARTING_POINTS)

    elevation_conditions = {p: random.choice(["sea", "land", "coast", "shallow"]) for p in icosa_usps_with_conditions}
    elevation_condition_functions = {
        "sea": lambda x: x < 0,
        "land": lambda x: x > 0,
        "coast": lambda x: abs(x) < 15,
        "shallow": lambda x: -5 <= x < 0,
    }
    data_points = icosa_usps_with_data  # can try doing even more points inside this, e.g. get conditions for only 6 iterations but generate data on 7

    # noise generation subject to the condition functions
    elevations = {p: 0 for p in data_points}
    is_frozen = {p: False for p in data_points}  # just hacking something up, random walk each point until it meets its criteria
    for i in range(10000):
        p = random.choice(data_points)
        if is_frozen[p]:
            continue
        elevations[p] += random.random()
        if p in elevation_conditions:
            condition_function = elevation_condition_functions[elevation_conditions[p]]
            if condition_function(elevations[p]):
                is_frozen[p] = True

    # now get the data and interpolate to plot
    data_coords = [p.latlondeg() for p in data_points]
    values = [elevations[p] for p in data_points]
    lat_range = [min_lat, max_lat]
    lon_range = [min_lon, max_lon]
    n_lats = 100
    n_lons = 200
    pu.plot_interpolated_data(data_coords, values, lat_range, lon_range, n_lats, n_lons, with_axis=True)
    # plt.scatter(icosa_lons, icosa_lats)
    plt.show()


if __name__ == "__main__":
    # test_generate_whole_planet()
    test_generate_on_section_of_condition_data()
    print("done")
