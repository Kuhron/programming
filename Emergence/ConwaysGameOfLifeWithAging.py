# like Conway's game of life, but still-lifes will die after N generations


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import sys
import time
import EmergenceMathUtil as emu


def create(max_age, plot_ion=False):
    # dr = 1
    # r_power = 0
    # r_to_power = emu.get_r_to_power_array(dr, dr, r_power)

    def evolve(field, max_age):
        # convolution = convolve(field, r_to_power, mode="same")
        neighbor_sum = emu.get_neighbor_nonzero_sum(field)
        ns = neighbor_sum

        assert (field < 0).sum() == 0, "negative field cells"

        alive_mask = field != 0
        dead_mask = field == 0
        too_old_mask = field >= max_age

        new_field = np.zeros(field.shape).astype(int)  # need astype(int) to ensure correct Life behavior
        # born if 3 neighbors
        # stay alive if 2 or 3 neighbors
        new_alive_mask = (
            (dead_mask & (ns == 3)) |  # note to self: if you write e.g. `dead_mask & ns == 3`, the & will be processed first, not the ==
            (alive_mask & ((ns == 2) | (ns == 3)))
        )
        new_alive_mask &= (~too_old_mask)  # even if it would live again, kill it if it's going to exceed max age once you add 1
        new_dead_mask = ~new_alive_mask

        # copy old values, add one for those that are alive this generation (0 > 1 for new alive, so don't need to condition on whether they were alive last generation)
        new_field[:] = field[:]
        new_field[new_dead_mask] = 0
        new_field[new_alive_mask] += 1

        return new_field

    def plot_field(field):
        plt.gcf().clear()
        # cmap = plt.cm.Spectral
        # cmap[0] = "black"
        plt.imshow(field)
        max_color_number = 1
        # im.set_over("yellow")
        plt.clim(0, max_color_number)
        plt.colorbar()
        plt.draw()

    space_shape = (100,100)  # for some reason (33,33) works for game of life but (34,34) and bigger (n,n) no longer display correct behavior! this turned out to be because np.zeros creates float, not int; somehow this resulted in changing the rules of Life when it was a certain size or larger, not sure why
    field = np.random.randint(0,2,space_shape)

    if plot_ion:
        plt.ion()
        fignum = plt.gcf().number  # use to determine if user has closed plot
        plot_field(field)
    for i in range(1000):
        if plot_ion:
            if not plt.fignum_exists(fignum):
                print("user closed plot; exiting")
                break

        field = evolve(field, max_age)
        if plot_ion:
            plot_field(field)
            plt.pause(0.05)

    # a last step, post-processing to reveal structure, if desired
    # field = emu.sigmoid(field)

    if plot_ion:
        plt.ioff()

    plot_field(field)
    plt.show()


if __name__ == "__main__":
    try:
        max_age = int(sys.argv[1])
    except IndexError:
        raise IndexError("need sys arg for max age")
    create(plot_ion=True, max_age=max_age)
