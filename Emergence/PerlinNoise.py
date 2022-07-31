import random
import numpy as np
import matplotlib.pyplot as plt

from noise.perlin import SimplexNoise


def get_2d_noise(min_val, max_val, x_range, y_range, n_xs, n_ys):
    # x_range and y_range affect how many oscillations of noise you'll get (bigger box = more oscillations)
    # n_xs and n_ys gives the shape of the returned array

    period = 2 * max(x_range, y_range)  # make sure it's not periodic within the square we are using
    sn = SimplexNoise(period=period)
    sn.randomize()
    noise_2d= np.vectorize(sn.noise2)

    xs = np.linspace(0, x_range, n_xs)
    ys = np.linspace(0, y_range, n_ys)
    X, Y = np.meshgrid(xs, ys)
    Z = noise_2d(X, Y)

    a, b = -1, 1  # default min and max, scale everything based on these
    if min_val == a and max_val == b:
        return Z
    else:
        # don't scale it based on the actual min and max that came up in the array, but based on the theoretical min and max (which are -1 and 1)
        total_range = b - a
        diff_above_min = Z - a
        alpha = diff_above_min / total_range
        assert (0 <= alpha).all() and (alpha <= 1).all()
        new_range = max_val - min_val
        new_Z = min_val + (alpha * new_range)
        return new_Z


if __name__ == "__main__":
    Z = get_2d_noise(min_val=0, max_val=1, x_range=20, y_range=12, n_xs=200, n_ys=120)
    plt.imshow(Z)
    plt.colorbar()
    plt.show()
