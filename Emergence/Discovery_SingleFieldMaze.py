# place to put functions which will run the whole process of creating some interesting emergent behavior
# for now, just paste the functions used as sub-functions into the body of a big function which runs the whole behavior of running a certain emergent process

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import time
import EmergenceMathUtil as emu


def create_single_field_maze(plot_ion=False):
    # save computation by just getting this kernel once
    dr = 1
    r_power = 0
    r_to_power = emu.get_r_to_power_array(dr, dr, r_power)

    def evolve(field):
        convolution = convolve(field, r_to_power, mode="same")
        c = convolution

        addition = 0  # initialize so rest of lines can all be +=
        addition += c - c**3  # x-x^3 will reinforce small deviations from 0 but punish large ones; increase coefficient on x to get bigger reinforcing range

        # limit addition to prevent exploding values
        addition = np.maximum(-10, np.minimum(10, addition))

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
    for i in range(100):
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
    else:
        # only plot at the end
        plot_field(field)
        plt.show()


if __name__ == "__main__":
    create_single_field_maze()
