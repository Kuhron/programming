from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
import NoiseMath as nm
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    edge_length_km = None
    iterations = 5
    test_lattice = IcosahedralGeodesicLattice(edge_length_km=edge_length_km, iterations=iterations)
    # test_lattice.plot_points()

    # memory profiling
    # objgraph.show_most_common_types(limit=20)
    # while True:
    #     typename = input("input object type to profile, or press enter to continue with program: ").strip()
    #     if typename == "":
    #         break
    #     obj = objgraph.by_type(typename)
    #     objgraph.show_backrefs([obj], max_depth=10)

    # making example images and data for each type of noise generation function
    df = test_lattice.create_dataframe()
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
    test_lattice.plot_data(df, "elevation", equirectangular=True, save=True, size_inches=(48, 24))

    plt.show()
    print("done")
