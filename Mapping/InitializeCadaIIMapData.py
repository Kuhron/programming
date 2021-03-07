from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
import NoiseMath as nm
import matplotlib.pyplot as plt
import os



if __name__ == "__main__":
    target_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/CadaIIMapData/"
    dir_is_empty = len(os.listdir(target_dir)) == 0
    if not dir_is_empty:
        raise Exception("Warning! The target dir is not empty. Aborting.")

    lattice = IcosahedralGeodesicLattice(iterations=7)  # max iterations before running out of memory
    df = lattice.create_dataframe()
    df["elevation"] = 0
    print(df)
    
    # add some random noise so plotting doesn't fail
    df = nm.add_random_data_independent_all_points(df, "elevation", n_iterations=1, sigma=0.1)
    lattice.plot_data(df, "elevation", equirectangular=True, size_inches=(48, 24))
    plt.show()

    output_fp = os.path.join(target_dir, "test.csv")
    lattice.write_data(df, output_fp)
