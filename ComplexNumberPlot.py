import matplotlib.pyplot as plt
import numpy as np


def f(z):
    # c = complex(-1.5, 0)
    c = get_random_complex_number_with_magnitude(0.4)
    return z ** 2 + c  # mandelbrot
    # return z * complex(np.sqrt(2) / 2, np.sqrt(2) / 2) * 1.01


def get_iterated_values(z0):
    z = z0
    while True:
        yield z
        z = f(z)


def get_random_complex_number_on_unit_circle():
    return get_random_complex_number_with_magnitude(1)


def get_random_complex_number_with_magnitude(r):
    theta = np.random.uniform(0, 2 * np.pi)
    return complex(r * np.cos(theta), r * np.sin(theta))


if __name__ == "__main__":
    # z0 = get_random_complex_number_on_unit_circle()
    z0 = complex(0, 0)

    g = get_iterated_values(z0)

    plt.ion()
    fignum = plt.gcf().number  # use to determine if user has closed plot

    GRAY = "#AAAAAA"
    plt.axhline(0, color=GRAY)
    plt.axvline(0, color=GRAY)

    for a in [-1, 1]:
        xs = np.arange(-1, 1, 0.01)
        ys = a * np.sqrt(1 - xs ** 2)
        xs = list(xs) + [1]  # np.arange won't cooperate with upper end 1.01
        ys = list(ys) + [0]
        plt.plot(xs, ys, c=GRAY)

    is_z0 = True
    for z in g:
        if not plt.fignum_exists(fignum):
            print("user closed plot; exiting")
            break
        elif np.absolute(z) > 1e100:
            print("sequence diverged too much")
            break
        else:
            if is_z0:
                plt.scatter(z.real, z.imag, c="k", alpha=1)
                is_z0 = False
            else:
                plt.scatter(z.real, z.imag, c="r", alpha=0.3)
            plt.draw()
            plt.pause(0.001)