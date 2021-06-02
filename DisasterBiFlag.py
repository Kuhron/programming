import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageColor
from scipy.signal import convolve2d


BI_COLORS_HEX = ["#D00070", "#8C4799", "#0032A0"]
# alternatively, ["#D70270", "#734F96", "#0038A8"]
BI_COLORS_RGB = [ImageColor.getcolor(x, "RGB") for x in BI_COLORS_HEX]
print(BI_COLORS_RGB)


def get_3state_noise(n_rows, n_cols):
    return np.random.randint(0, 3, (n_rows, n_cols))  # np randint is right-exclusive


def get_stripes(n_rows, n_cols, mode):
    col_nums, row_nums = np.meshgrid(range(n_cols), range(n_rows))  # stupid transpose

    a,b,c,d = np.random.normal(0, 0.01, (4,))
    warp_func = np.vectorize(lambda x: np.sin(a*x+b) + np.cos(c*x+d))

    if mode == "horizontal":
        arr = warp_func(row_nums)
        return (arr % 3).astype(int)
    elif mode == "vertical":
        arr = warp_func(col_nums)
        return (arr % 3).astype(int)
    elif mode == "diagonal":
        arr = warp_func(row_nums + col_nums)
        return (arr % 3).astype(int)
    else:
        raise ValueError(f"unknown mode {mode}")


def get_3state_evolved(n_rows, n_cols, plot=True):
    # start with noise, evolve it according to some cellular automaton rule
    n_steps = 100000
    arr = get_3state_noise(n_rows, n_cols)
    # arr = get_stripes(n_rows, n_cols, mode="diagonal")

    # convolution_arr = np.random.randint(-2, 3, (3,3))  # this is fixed throughout
    # convolution_arr = np.random.uniform(-1, 1, (3,3))  # as long as cast to int after convolution, this may help get better dynamics
    # convolution_arr = np.array([[1,1,1], [1,0,1], [1,1,1]])  # D8 neighbors
    # convolution_arr = 1/4 * np.array([[1,1,1], [1,0,1], [1,1,1]])  # THIS IS GOOD!
    convolution_arr = 1/5.3 * np.array([[1,1,1], [1,0,1], [1,1,1]])  # this constant multiplier has a critical point below which the system will die out because it can't reach 3 and cycle back (1/6 is too low, 1/5 is still dynamic); # 1/5.5 has cool stable behavior; 1/5.2 has cool slow decay; sharp phase transition between 1/5 and 1/5+epsilon; 
    # convolution_arr = np.array([[0,1,0], [1,0,1], [0,1,0]])  # D4 neighbors
    # convolution_arr = np.array([[1,0,1], [0,0,0], [1,0,1]])  # diagonals only
    # convolution_arr = np.zeros((3,3))  # test case, all zero
    # convolution_arr = np.array([[0,0,0], [0,1,0], [0,0,0]])  # test case, identity
    print("convolution arr:")
    print(convolution_arr)

    if plot:
        plt.ion()
        fignum = plt.gcf().number  # use to determine if user has closed plot
    for step_i in range(n_steps):
        arr = convolve2d(arr, convolution_arr, mode="same", boundary="wrap")
        arr += get_stripes(n_rows, n_cols, mode="horizontal")
        arr = np.mod(arr, 3).astype(int)

        if plot and step_i % 1 == 0:
            if plot and not plt.fignum_exists(fignum):
                print("user closed plot")
                break
            plt.gcf().clear()
            plt.imshow(arr)
            plt.title(f"step {step_i}")
            plt.draw()
            plt.pause(0.01)

    if plot:
        plt.ioff()
        plt.show()

    return arr


def get_design(n_rows, n_cols):
    # return get_completely_random_design(n_rows, n_cols)
    index_arr = get_3state_evolved(n_rows, n_cols)
    return translate_indices_to_rgb(index_arr)


def translate_indices_to_rgb(index_arr):
    rgb_first_form = np.vectorize(lambda i: BI_COLORS_RGB[i])(index_arr)
    rgb_last_form = np.moveaxis(rgb_first_form, 0, -1)
    return rgb_last_form


def get_completely_random_design(n_rows, n_cols):
    # np.random.choice doesn't like options with multiple dimensions (even if it's a 1D array of immutable iterables)
    arr = [[random.choice(BI_COLORS_RGB) for c in range(n_cols)] for r in range(n_rows)]
    return np.array(arr)


def plot_design(arr):
    plt.gcf().tight_layout()
    plt.imshow(arr)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    n_rows = 90
    n_cols = int(3/2 * n_rows)
    arr = get_design(n_rows, n_cols)
    plot_design(arr)
