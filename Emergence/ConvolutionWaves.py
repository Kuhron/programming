import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


# idea: convolving random noise N times is sort of like having waves of wavelength N
# so multiple convolved arrays like this can be added to get a spectrum of 2d noise with different power at different frequencies


def get_noise(shape, amplitude, n_convolutions):
    arr = np.random.uniform(-amplitude, amplitude, shape)

    convolution = np.array([[1,1,1],[1,0,1],[1,1,1]])
    for i in range(n_convolutions):
        arr = convolve2d(arr, convolution, mode="same", boundary="wrap")
    return arr


def get_spectral_noise(shape, spectrum):
    arr = np.zeros(shape)
    for n_convolutions, amplitude in spectrum.items():
        this_component = get_noise(shape, amplitude, n_convolutions)
        arr += this_component
    return arr


if __name__ == "__main__":
    # spectrum = {3*n: 1/np.log(n+1) for n in range(1, 10)}
    spectrum = {1: 100, 2: 50, 3: 20, 4: 35, 5: 10, 8: 15, 11: 5}
    shape = (360, 480)
    arr = get_spectral_noise(shape, spectrum)
    plt.imshow(arr)
    # plt.contourf(arr, levels=30)
    plt.contour(arr, levels=10, colors="k")
    plt.show()
