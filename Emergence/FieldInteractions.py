# trying to get interesting emergent properties from some dynamical system which isn't that complicated
# more is different
# might not be feasible to simulate it on the scale that would be required for emergence, but who knows, little unexpected things are also worth it (e.g. cellular automaton order/chaos coexistenece is easy to simulate)


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import time
import EmergenceMathUtil as emu


def evolve(red_field, yellow_field, blue_field):
    red_field_1_over_r2 = emu.get_power_of_r_sums(red_field, power=-2)
    yellow_field_1_over_r2 = emu.get_power_of_r_sums(yellow_field, power=-2)
    blue_field_1_over_r2 = emu.get_power_of_r_sums(blue_field, power=-2)

    red_field_addition = yellow_field_1_over_r2 - blue_field_1_over_r2
    yellow_field_addition = blue_field_1_over_r2 - red_field_1_over_r2
    blue_field_addition = red_field_1_over_r2 - yellow_field_1_over_r2
    # red_field_addition = 0
    # yellow_field_addition = 0
    # blue_field_addition = 0

    red_field += red_field_addition
    yellow_field += yellow_field_addition
    blue_field += blue_field_addition

    # correct for highly divergent values AFTER creating them (so you don't get diverging oscillation when the addition and correction are in the same direction)
    # how to do this mathematically without overshooting the correction?
    # could model after Hooke's Law or something, restoring force is just proportional to the displacement
    correction_rate = 0.01
    correction_power = 1
    red_field_sign = np.sign(red_field)
    yellow_field_sign = np.sign(yellow_field)
    blue_field_sign = np.sign(blue_field)
    red_field_correction = -1 * red_field_sign * correction_rate * abs(red_field**correction_power)
    yellow_field_correction = -1 * yellow_field_sign * correction_rate * abs(yellow_field**correction_power)
    blue_field_correction = -1 * blue_field_sign * correction_rate * abs(blue_field**correction_power)

    red_field += red_field_correction
    yellow_field += yellow_field_correction
    blue_field += blue_field_correction

    return (red_field, yellow_field, blue_field)


def plot_fields(red_field, yellow_field, blue_field):
    plt.gcf().clear()
    plt.subplot(1,3,1)
    plt.imshow(red_field)
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.imshow(yellow_field)
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(blue_field)
    plt.colorbar()

    plt.draw()


if __name__ == "__main__":
    space_shape = (200, 200)
    red_field = np.random.normal(0, 1, space_shape)
    yellow_field = np.random.normal(0, 1, space_shape)
    blue_field = np.random.normal(0, 1, space_shape)

    plt.ion()
    fignum = plt.gcf().number  # use to determine if user has closed plot
    plot_fields(red_field, yellow_field, blue_field)
    for i in range(100):
        if not plt.fignum_exists(fignum):
            print("user closed plot; exiting")
            break

        red_field, yellow_field, blue_field = evolve(red_field, yellow_field, blue_field)
        plot_fields(red_field, yellow_field, blue_field)
        plt.pause(0.05)
    plt.ioff()
    plt.show()
