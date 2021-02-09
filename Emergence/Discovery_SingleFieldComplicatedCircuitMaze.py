import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import time
import EmergenceMathUtil as emu


def create(plot_ion=False):
    def evolve(field):
        dr = 6
        r_power = -1
        r_to_power = emu.get_r_to_power_array(dr, dr, r_power)
        convolution = convolve(field, r_to_power, mode="same")
        c = convolution

        addition = 0  # initialize so rest of lines can all be +=
        # addition += -0.1 * c / np.mean(c)
        # addition += np.random.normal(0, 1, field.shape)
        addition += (5**2)*c - c**3  # x-x^3 will reinforce small deviations from 0 but punish large ones; increase coefficient on x to get bigger reinforcing range

        # limit addition to prevent exploding values
        # addition = emu.signed_log(addition)
        addition = np.sign(addition)

        field += addition
        return field

    def plot_field(field):
        plt.gcf().clear()
        plt.imshow(field)
        plt.colorbar()
        plt.draw()

    space_shape = (200, 200)
    field = np.random.normal(0, 1, space_shape)

    if plot_ion:
        plt.ion()
        fignum = plt.gcf().number  # use to determine if user has closed plot
        plot_field(field)
    for i in range(200):
        if plot_ion:
            if not plt.fignum_exists(fignum):
                print("user closed plot; exiting")
                break

        field = evolve(field)
        if plot_ion:
            plot_field(field)
            plt.pause(0.05)
    if plot_ion:
        plt.ioff()
        plt.show()
    else:
        plot_field(field)
        plt.show()


if __name__ == "__main__":
    create(plot_ion=True)
