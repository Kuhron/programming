import random
import numpy as np
import matplotlib.pyplot as plt


def get_random_2d_displacement(shape):
    x, y = shape
    max_dx = x//2  # if it has extra half, still can't go that far
    max_dy = y//2
    return random.randint(0,max_dx), random.randint(0,max_dy)


def get_all_2d_displacements(shape):
    x,y = shape
    max_dx = x//2
    max_dy = y//2
    for dx in range(max_dx):
        for dy in range(max_dy):
            yield (dx,dy)


def get_autocorrelation(m, v):
    dx, dy = v
    mx, my = m.shape
    n_rows_to_get = mx-dx
    n_cols_to_get = my-dy
    m1 = m[:n_rows_to_get, :n_cols_to_get]
    m2 = m[-n_rows_to_get:, -n_cols_to_get:]
    m1 = m1.flatten()
    m2 = m2.flatten()  # make them 1d for easier correlation
    ac = np.corrcoef(m1, m2)  # np.correlate() is something else
    assert ac.shape == (2,2)
    return ac[0,1]  # corr of m1 with m2 elementwise


if __name__ == "__main__":
    m = np.random.uniform(-1,1,(30,30))
    m[1:,:] += m[:-1,:]  # add some artificial autocorrelation in one direction
    m[2:,:] += m[:-2,:]
    m[:,5:] += m[:,:-5]
    # displacement_vector = get_random_2d_displacement(m.shape)
    z = np.zeros(m.shape)
    for displacement_vector in get_all_2d_displacements(m.shape):
        if displacement_vector != (0,0):
            print(displacement_vector)
            ac = get_autocorrelation(m, displacement_vector)
            print(ac)
            dx, dy = displacement_vector
            z[dx][dy] = ac
    plt.imshow(z)
    plt.colorbar()
    plt.show()
