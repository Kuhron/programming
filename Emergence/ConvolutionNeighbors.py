import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import time
import EmergenceMathUtil as emu
from InteractivePlot import InteractivePlot



if __name__ == "__main__":
    # arr = np.random.randint(0, 2, (100, 100))
    arr = np.random.normal(0, 1, (100, 100))
    arr -= np.mean(arr)  # set mean to 0

    plot_every_n_steps = 1
    with InteractivePlot(plot_every_n_steps=plot_every_n_steps) as iplt:
        while iplt.is_open():
            plt.gcf().clear()
            plt.imshow(arr)
            plt.colorbar()
            plt.draw()
            plt.pause(0.01)

            # arr = emu.get_neighbor_sum(arr)
            arr = emu.get_power_of_r_sums(arr, power=-2)

    plt.ioff()
    plt.show()
