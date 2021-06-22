# original author: Lucas Brown
#Adv. GIS final project
#5/30/19

#random interpolated surface generator

import random
import numpy as np
import matplotlib.pyplot as plt


def get_noise(grid_size, variability):
    # grid_size is length on a side of the square grid
    grid = np.zeros((grid_size, grid_size))

    grid[0,0] = random.uniform(0, 10)
    v = variability
    for row_i in range(0, grid_size):
        if row_i == 0:
            # we already set cell [0,0] to the seed value
            pass
        else:
            first_value_in_previous_row = grid[row_i-1, 0]
            m = first_value_in_previous_row
            val = random.uniform(m - v, m + v)
            grid[row_i, 0] = val

        # col_i = 0 has already been set
        for col_i in range(1, grid_size):
            if row_i == 0:
                previous_value_in_row = grid[row_i, col_i-1]
                m = previous_value_in_row
                val = random.uniform(m - v, m + v)   #for the first column: add points based on the point that came before.
                grid[row_i, col_i] = val  # this keeps the raster from being too random and unable to interpolate properly
            else:
                val_to_left = grid[row_i, col_i-1]
                val_above = grid[row_i-1, col_i]
                m = 1/2 * (val_to_left + val_above)
                dev = 0.75 * v
                val = random.uniform(m - dev, m + dev)
                grid[row_i, col_i] = val  # variation is decreased in these points compared to the others due to fewer degrees of freedom

    return grid


if __name__ == "__main__":
    grid_size = 100
    variability = 3
    arr = get_noise(grid_size, variability)
    plt.imshow(arr)
    plt.colorbar()
    plt.show()


