import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import time
import EmergenceMathUtil as emu


def create(plot_ion=False):
    # dr = 1
    # r_power = 0
    # r_to_power = emu.get_r_to_power_array(dr, dr, r_power)

    def evolve(field):
        # convolution = convolve(field, r_to_power, mode="same")
        neighbor_sum = emu.get_neighbor_sum(field)
        ns = neighbor_sum
        # c = neighbor_sum

        # addition = 0  # initialize so rest of lines can all be +=
        # addition += -0.1 * c / np.mean(c)
        # addition += np.random.normal(0, 1, field.shape)
        # addition += (5**2)*c - c**3  # x-x^3 will reinforce small deviations from 0 but punish large ones; increase coefficient on x to get bigger reinforcing range

        # limit addition to prevent exploding values
        # addition = emu.signed_log(addition)
        # addition = np.sign(addition)
        # addition = np.where(abs(addition) > 2, 0, np.sign(addition))  # might get interesting holes of 0 in the function this way (around critical points)

        alive_mask = field == 1
        dead_mask = field == 0
        # mask_2 = neighbor_sum == 2
        # mask_3 = neighbor_sum == 3

        new_field = np.zeros(field.shape).astype(int)  # need astype(int) to ensure correct Life behavior
        # born if 3 neighbors
        # stay alive if 2 or 3 neighbors
        new_alive_mask = (
            (dead_mask & (ns == 3)) |  # note to self: if you write e.g. `dead_mask & ns == 3`, the & will be processed first, not the ==
            (alive_mask & ((ns >= 3) & (ns <= 4)))
        )
        new_field[new_alive_mask] = 1

        # old way, this works
        # new_field[dead_mask & mask_3] = 1
        # new_field[alive_mask & (mask_2 | mask_3)] = 1

        return new_field

    def plot_field(field):
        plt.gcf().clear()
        plt.imshow(field)
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

        field = evolve(field)
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
    create(plot_ion=True)
