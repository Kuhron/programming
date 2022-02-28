import numpy as np
import matplotlib.pyplot as plt


def f_mandelbrot(z, current_value):
    return current_value**2 + z


def hue_mandelbrot(current_value):
    # is the z likely to be in the orbit set, based on current value, if so give NaN
    a = abs(current_value)
    distance_from_1 = a - 1
    res = np.zeros(distance_from_1.shape)
    res[distance_from_1 <= 0] = np.nan
    res[distance_from_1 > 0] = np.log(distance_from_1)
    # for stuff outside the set, the hue should rotate based on how far away it is from 1, log scale
    return res % 1


def iterate_f(f, z, initial_value, n_times):
    res = []
    current_value = initial_value
    for i in range(n_times + 1):
        res.append(current_value)
        current_value = f(z, current_value)
    return res


def plot_set(f, hue_f):
    xs = np.linspace(-2, 2, 10)
    ys = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(xs, ys)
    Z = X + Y*1j
    iterations = iterate_f(f, Z, initial_value=0, n_times=100)
    last_iteration = iterations[-1]
    # abs_last_iteration = abs(last_iteration)  # this DOES work for magnitude of complex number
    hues = hue_f(last_iteration)
    print(hues)
    cmap = plt.get_cmap("hsv")  # cyclic through saturated hues
    plt.imshow(hues, cmap=cmap)
    plt.show()


if __name__ == "__main__":
    plot_set(f=f_mandelbrot, hue_f=hue_mandelbrot)
