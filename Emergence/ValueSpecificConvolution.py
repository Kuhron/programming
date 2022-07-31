import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


class ConvolutionByValue:
    def __init__(self):
        self.dct = {}

    def __getitem__(self, index):
        if index not in self.dct:
            self.dct[index] = ConvolutionByValue.get_random_kernel()
            if len(self.dct) % 1000 == 0 and len(self.dct) != 0:
                print(f"warning: {len(self.dct)} entries reached")
        return self.dct[index]

    @staticmethod
    def get_random_kernel():
        return np.random.randint(0, 2, (3, 3))


def convolve_by_value(arr, kernels, convolve_values=False, condition_function=None):
    # condition function gets a value corresponding to a condition, e.g. the values map instead to (val % 2) so all cells which are even will get the convolution for value of 0, and all cells which are odd will get a different convolution kernel
    assert type(convolve_values) is bool  # passing later kwarg as positional

    values = np.unique(arr)
    res = np.zeros(arr.shape)
    if condition_function is None:
        condition_function = lambda x: x
    for val in values:
        mask = condition_function(arr) == val
        arr_to_convolve = np.where(mask, arr, 0) if convolve_values else mask
        addend = convolve2d(arr_to_convolve, kernels[val], mode="same", boundary="wrap")
        res += addend
    return res


def get_single_point_arr(shape):
    arr = np.zeros(shape)
    nx, ny = shape
    x = nx//2
    y = ny//2
    arr[x,y] = 1
    return arr


if __name__ == "__main__":
    shape = (300,400)
    arr = np.random.randint(0, 10, shape)
    # arr = get_single_point_arr(shape)
    convolution_by_value = ConvolutionByValue()

    plt.ion()
    fignum = plt.gcf().number

    while True:
        if not plt.fignum_exists(fignum):
            plt.ioff()
            plt.imshow(arr)
            plt.colorbar()
            plt.show()
            break
        plt.gcf().clear()
        plt.imshow(arr)
        plt.colorbar()
        plt.draw()
        plt.pause(0.01)

        # condition_function = None
        # condition_function = lambda x: x % 2
        # condition_function = lambda x: x % 3
        # condition_function = lambda x: abs(5-x)
        # condition_function = lambda x: 9-x
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        condition_function = lambda x: x**a % b

        arr = convolve_by_value(arr, convolution_by_value, condition_function=condition_function)
        if len(np.unique(arr)) == 1:
            print("degenerate state reached")
            input("press enter to exit")
            break


