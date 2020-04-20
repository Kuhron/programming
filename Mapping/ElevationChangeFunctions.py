import numpy as np


def elevation_change_parabolic(d, max_d, max_change):
    b = max_d
    h = max_change
    return -h/(b**2) * (d**2) + 2*h/b * d

def elevation_change_linear(d, max_d, max_change):
    b = max_d
    h = max_change
    return h/b * d

def elevation_change_semicircle(d, max_d, max_change):
    b = max_d
    h = max_change
    return h/b * np.sqrt(2*b*d - d**2)

def elevation_change_inverted_semicircle(d, max_d, max_change):
    b = max_d
    h = max_change
    return h - h/b * np.sqrt(b**2 - d**2)

def elevation_change_sinusoidal(d, max_d, max_change):
    b = max_d
    h = max_change
    return h/2 * (1 + np.sin(np.pi/b * (d - b/2)))

def elevation_change_constant(d, max_d, max_change):
    return max_change

ELEVATION_CHANGE_FUNCTIONS = [
    elevation_change_linear,  # default to linear, should be fastest non-constant
    # elevation_change_parabolic,
    # elevation_change_semicircle,  # changes too fast near 0, makes very steep slopes too often
    # elevation_change_inverted_semicircle,  # slow
    # elevation_change_sinusoidal,  # slow
    # elevation_change_constant,  # f(0) is not 0, makes cliffs
]

def show_elevation_change_functions():
    xs = np.linspace(0, 1, 20)
    max_x = 1
    max_change = 1
    for i, f in enumerate(ELEVATION_CHANGE_FUNCTIONS):
        ys = [f(x, max_x, max_change) for x in xs]
        plt.plot(xs, ys)
        plt.title("function index {}".format(i))
        plt.show()
