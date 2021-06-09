import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def get_noise(n_rows, n_cols, modulus):
    return np.random.uniform(0, modulus, (n_rows, n_cols))


def evolve_ca(n_rows, n_cols, modulus, neighborhood_type, neighbor_sum_constant, modification_function, plot=True):
    print(locals())

    n_steps = 100000
    arr = get_noise(n_rows, n_cols, modulus)
    
    if neighborhood_type == "D4":
        convolution_arr = np.array([[0,1,0],[1,0,1],[0,1,0]])
    elif neighborhood_type == "D8":
        convolution_arr = np.array([[1,1,1],[1,0,1],[1,1,1]])
    elif neighborhood_type == "diagonal":
        convolution_arr = np.array([[1,0,1],[0,0,0],[1,0,1]])
    else:
        raise ValueError(f"unknown neighborhood type: {neighborhood_type}")

    convolution_arr = convolution_arr * neighbor_sum_constant

    # modification function maps the interval [0,m] to itself in some way
    if modification_function == "id":
        mfunc = lambda x: x
    elif modification_function == "floor":
        mfunc = lambda x: x.astype(int)
    elif modification_function == "parabola":
        # parabolic bump centered on [0,m] with peak at m/2
        mfunc = lambda x, m=modulus: (x*(m-x)) * m / (m**2 / 4)  # normalize so the peak goes all the way up to the modulus
    elif modification_function == "triangle":
        # triangular bump centered on [0,m] with peak at m/2
        mfunc = lambda x, m=modulus: m - abs(2*x - m)
    else:
        raise ValueError(f"unknown modification function {modification_function}")

    if plot:
        plt.ion()
        fignum = plt.gcf().number  # use to determine if user has closed plot
    for step_i in range(n_steps):
        arr = convolve2d(arr, convolution_arr, mode="same", boundary="wrap")

        arr = np.mod(arr, modulus)
        arr = mfunc(arr)

        if plot and step_i % 1 == 0:
            if plot and not plt.fignum_exists(fignum):
                print("user closed plot")
                break
            plt.gcf().clear()
            plt.imshow(arr)
            plt.colorbar()
            plt.title(f"step {step_i}")
            plt.draw()
            plt.pause(0.01)

    if plot:
        plt.ioff()
        plt.show()

    return arr


if __name__ == "__main__":
    n_rows = 240
    n_cols = int(3/2 * n_rows)
    modulus = 1 + abs(np.random.normal(0, 2))
    neighborhood_type = random.choice(["D8", "D4", "diagonal"])
    # neighborhood_type = "D8"
    neighbor_sum_constant = np.random.normal(0, 1)
    modification_function = random.choice(["id", "floor", "parabola", "triangle"])
    plot = True
    params = {"n_rows": n_rows, "n_cols": n_cols, "modulus": modulus, "neighborhood_type": neighborhood_type, "neighbor_sum_constant": neighbor_sum_constant, "modification_function": modification_function, "plot": plot}

    # can put discoveries here, have the evolve function print locals()
    fuzzy_cells_and_prions = {'modulus': 3.6184508776628803, 'neighborhood_type': 'D8', 'neighbor_sum_constant': -0.1324512456707755, 'modification_function': 'id'}
    flashing_amoebas = {'modulus': 1.4229596110653309, 'neighborhood_type': 'D8', 'neighbor_sum_constant': -0.1495842896627948, 'modification_function': 'triangle'}
    devolution_to_braille = {'modulus': 1.7220424043870777, 'neighborhood_type': 'diagonal', 'neighbor_sum_constant': 0.7087672802040939, 'modification_function': 'floor'}
    wobbly_oscillator = {'modulus': 4.982426272634422, 'neighborhood_type': 'D4', 'neighbor_sum_constant': -0.13755029176642564, 'modification_function': 'triangle'}

    # for re-running a discovery
    params.update(wobbly_oscillator)

    arr = evolve_ca(**params)
    plt.imshow(arr)
    plt.show()
