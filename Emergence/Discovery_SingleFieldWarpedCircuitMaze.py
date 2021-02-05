import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import time
import EmergenceMathUtil as emu


def evolve(field):
    dr = 3
    r_power = 0
    r_to_power = emu.get_r_to_power_array(dr, dr, r_power)
    convolution = convolve(field, r_to_power, mode="same")
    c = convolution

    addition = 0  # initialize so rest of lines can all be +=
    # addition += -0.1 * c / np.mean(c)
    # addition += np.random.normal(0, 1, field.shape)
    addition += c - c**3  # x-x^3 will reinforce small deviations from 0 but punish large ones; increase coefficient on x to get bigger reinforcing range

    # limit addition to prevent exploding values
    addition = emu.signed_log(addition)

    field += addition
    return field


def plot_field(field):
    plt.gcf().clear()
    plt.imshow(field)
    plt.colorbar()
    plt.draw()


if __name__ == "__main__":
    space_shape = (200, 200)
    field = np.random.normal(0, 1, space_shape)

    plt.ion()
    fignum = plt.gcf().number  # use to determine if user has closed plot
    plot_field(field)
    for i in range(100):
        if not plt.fignum_exists(fignum):
            print("user closed plot; exiting")
            break

        field = evolve(field)
        plot_field(field)
        plt.pause(0.05)
    plt.ioff()
    plt.show()
