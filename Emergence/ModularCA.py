import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def get_noise(n_rows, n_cols, modulus):
    return np.random.uniform(0, modulus, (n_rows, n_cols))


def get_sparse_noise(n_rows, n_cols, modulus, freq):
    r_arr = np.random.random((n_rows, n_cols))
    has_value = r_arr < freq  # similar to if random.random() < 0.01
    values = get_noise(n_rows, n_cols, modulus)
    arr = np.where(has_value, values, 0)
    return arr


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def inverse_sigmoid(x):
    # map [0,1] to [-inf,inf] in order to do a domain-R function on the [0,1] interval
    return np.log(x/(1-x))


@np.vectorize  # since it has if statement
def sigmoid_in_01(x, distortion_constant):
    # distortion constant of 1 gives y=x (confined to [0,1])
    # c of > 1 gives s-curve where derivative at 0 and 1 is 0
    # 0 < c < 1 gives inverse-sigmoid-looking curve where derivative at 0 and 1 is inf
    # negative c is similar but flipped about x=0.5
    c = distortion_constant

    # avoid NaN problems on 0 and 1 themselves
    if x == 0:
        return 0
    elif x == 1:
        return 1
    # return sigmoid(c * inverse_sigmoid(x))
    # simplification of the expression:
    return 1/(1 + (x/(1-x))**-c)


def evolve_ca(initial_state, modulus, neighborhood_type, neighbor_sum_constant, modification_function, plot, plot_every_n_steps):
    params_to_print = {k:v for k,v in locals().items() if k not in ["initial_state", "plot", "plot_every_n_steps"]}
    print(params_to_print)

    n_steps = 100000
    arr = initial_state
    
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
    elif modification_function.startswith("sigmoid"):
        distortion_constant = float(modification_function.replace("sigmoid", ""))
        mfunc = lambda x, c=distortion_constant: sigmoid_in_01(x,c)
    else:
        raise ValueError(f"unknown modification function {modification_function}")

    if plot:
        plt.ion()
        fignum = plt.gcf().number  # use to determine if user has closed plot
    for step_i in range(n_steps):
        arr = convolve2d(arr, convolution_arr, mode="same", boundary="wrap")

        arr = np.mod(arr, modulus)
        arr = mfunc(arr)

        if plot and step_i % plot_every_n_steps == 0:
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
    n_rows = 120
    n_cols = int(3/2 * n_rows)
    modulus = 1 + abs(np.random.normal(0, 2))
    neighborhood_type = random.choice(["D8", "D4", "diagonal"])
    # neighborhood_type = "D8"
    neighbor_sum_constant = np.random.normal(0, 1)
    modification_function = random.choice(["id", "floor", "parabola", "triangle", "sigmoid2", "sigmoid5"])
    plot = True
    plot_every_n_steps = 1

    initial_state = get_noise(n_rows, n_cols, modulus)
    # initial_state = get_sparse_noise(n_rows, n_cols, modulus, freq=0.01)

    params = {"initial_state": initial_state, "modulus": modulus, "neighborhood_type": neighborhood_type, "neighbor_sum_constant": neighbor_sum_constant, "modification_function": modification_function, "plot": plot, "plot_every_n_steps": plot_every_n_steps}

    # can put discoveries here, have the evolve function print locals()
    fuzzy_cells_and_prions = {'modulus': 3.6184508776628803, 'neighborhood_type': 'D8', 'neighbor_sum_constant': -0.1324512456707755, 'modification_function': 'id'}
    flashing_amoebas = {'modulus': 1.4229596110653309, 'neighborhood_type': 'D8', 'neighbor_sum_constant': -0.1495842896627948, 'modification_function': 'triangle'}
    devolution_to_braille = {'modulus': 1.7220424043870777, 'neighborhood_type': 'diagonal', 'neighbor_sum_constant': 0.7087672802040939, 'modification_function': 'floor'}
    wobbly_oscillator = {'modulus': 4.982426272634422, 'neighborhood_type': 'D4', 'neighbor_sum_constant': -0.13755029176642564, 'modification_function': 'triangle'}  # becomes overlapping positive/negative blobs, with intervening checkerboard space, and the blobs slowly collapse to circles and disappear; fine structure within a color is also visible, curves perpendicular to border lines
    flowers_in_the_sea = {'modulus': 3.036957539883622, 'neighborhood_type': 'D4', 'neighbor_sum_constant': -0.30437791227925504, 'modification_function': 'floor'}
    flashing_growing_checkers = {'modulus': 3.9883087817089526, 'neighborhood_type': 'diagonal', 'neighbor_sum_constant': -0.9821112017744941, 'modification_function': 'sigmoid2'}
    jumping_across_barrier = {'modulus': 3.2041633924528496, 'neighborhood_type': 'D4', 'neighbor_sum_constant': 0.26478347096386534, 'modification_function': 'id'}
    slash_and_flash = {'modulus': 2.4356354632491204, 'neighborhood_type': 'D4', 'neighbor_sum_constant': -0.6230227015040277, 'modification_function': 'sigmoid2'}
    devolution_to_oscillating_dots = {'modulus': 2.551584048124375, 'neighborhood_type': 'D4', 'neighbor_sum_constant': -1.0385972513275994, 'modification_function': 'sigmoid2'}
    cpu_growth = {'modulus': 1.2393709512216025, 'neighborhood_type': 'D8', 'neighbor_sum_constant': 1.1596031460068366, 'modification_function': 'floor'}  # all the cool stuff is at the very beginning, good to start from sparse initial state to watch the crystals grow
    amoebic_rips = {'modulus': 1.135970110957478, 'neighborhood_type': 'D8', 'neighbor_sum_constant': -0.1542448467294696, 'modification_function': 'sigmoid2'}

    # for re-running a discovery
    # params.update(amoebic_rips)

    arr = evolve_ca(**params)
    plt.imshow(arr)
    plt.show()
