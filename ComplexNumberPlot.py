import matplotlib.pyplot as plt
import numpy as np


def f(z):
    return z ** 2
    # return z * complex(np.sqrt(2) / 2, np.sqrt(2) / 2) * 1.01


def get_iterated_values(z0):
    z = z0
    while True:
        yield z
        z = f(z)


if __name__ == "__main__":
    r = 1
    theta = np.random.uniform(0, 2 * np.pi)
    z0 = complex(r * np.cos(theta), r * np.sin(theta))
    g = get_iterated_values(z0)

    plt.ion()
    fignum = plt.gcf().number  # use to determine if user has closed plot

    for z in g:
        if plt.fignum_exists(fignum):
            plt.scatter(z.real, z.imag, c="r")
            plt.draw()
            plt.pause(0.01)
        else:
            print("user closed plot; exiting")
            break