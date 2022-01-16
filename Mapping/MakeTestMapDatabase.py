from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
from IcosahedronPointDatabase import IcosahedronPointDatabase
import IcosahedronMath as icm
import NoiseMath as nm
import matplotlib.pyplot as plt
import os
import random


def make_test_db():
    root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/TestMapData/"
    if IcosahedronPointDatabase.db_exists(root_dir):
        db = IcosahedronPointDatabase.load(root_dir)
        print("db loaded")
    else:
        block_size = 65536
        db = IcosahedronPointDatabase.new(root_dir, block_size)
        db.add_variable("is_land")
        db.add_variable("elevation")
        print("new db created")

    max_p_i = icm.get_points_from_iterations(7) - 1
    for i in range(1000000):
        p_i = random.randint(0, max_p_i)
        # print("elevation at point", p_i, "is", db[p_i, "elevation"])
        elevation = int(round(random.normalvariate(0, 100), 0))
        db[p_i, "elevation"] = elevation

        if i % 1000 == 1:
            print(i)
            db.write(clear_cache=True)

    db.write(clear_cache=True)


if __name__ == "__main__":
    make_test_db()
    raise NotImplementedError

    lattice = IcosahedralGeodesicLattice(iterations=6)  # max of 7 iterations before running out of memory
    df = lattice.create_dataframe()
    df["elevation"] = 0
    print(df)
    
    # add some random noise so plotting doesn't fail
    df = nm.add_random_data_independent_all_points(df, "elevation", n_iterations=1, sigma=0.1)
    lattice.plot_data(df, "elevation", equirectangular=True, size_inches=(48, 24))
    plt.show()

    output_fp = os.path.join(target_dir, "test.csv")
    lattice.write_data(df, output_fp)
