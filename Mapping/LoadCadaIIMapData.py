from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
import NoiseMath as nm
import matplotlib.pyplot as plt
import pandas as pd
import os



if __name__ == "__main__":
    target_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/CadaIIMapData/"
    input_file = "test.csv"  # later replace this with the actual database
    input_fp = os.path.join(target_dir, input_file)

    df = pd.read_csv(input_fp, index_col="index")
    n_rows = len(df.index)
    iterations = IcosahedralGeodesicLattice.get_iterations_from_number_of_points(n_rows)
    lattice = IcosahedralGeodesicLattice(iterations=iterations)
    lattice_df = lattice.create_dataframe()
    needed_columns = ["usp", "xyz", "latlondeg"]  # things left out of the written df
    df[needed_columns] = lattice_df[needed_columns]  # populate with values from the lattice computation

    lattice.plot_data(df, "elevation", equirectangular=True, size_inches=(48, 24))
    plt.show()

