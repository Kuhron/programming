import numpy as np
import matplotlib.pyplot as plt


# as much as possible, normalize given distance into [0, 1] by dividing by max_d, and function should have range [0, 1] on that interval
# then multiply output by max_change


def linear(d, max_d, max_change):
    x = d/max_d
    y = x
    return max_change * y

def parabolic_negative_acceleration(d, max_d, max_change):
    x = d/max_d
    y = 2*x - x**2
    return max_change * y

def parabolic_positive_acceleration(d, max_d, max_change):
    x = d/max_d
    y = x**2
    return max_change * y

def semicircle_negative_acceleration(d, max_d, max_change):
    x = d/max_d
    y = np.sqrt(2*x - x**2)  # interesting that this is the square root of the parabolic (negative acceleration) expression!
    return max_change * y

def semicircle_positive_acceleration(d, max_d, max_change):
    x = d/max_d
    y = 1 - np.sqrt(1 - x**2)
    return max_change * y

def sinusoidal(d, max_d, max_change):
    x = d/max_d
    y = 1/2 * (1 + np.sin(np.pi * (x - 1/2)))
    return max_change * y

def exponential(d, max_d, max_change):
    base = 20  # steepness
    x = d/max_d
    y = (base**x - 1)/(base - 1)
    return max_change * y

def constant(d, max_d, max_change):
    return max_change

ELEVATION_CHANGE_FUNCTIONS = {
    # function, spikiness
    linear: 0.5,
    parabolic_positive_acceleration: 0.75,
    parabolic_negative_acceleration: 0.25,
    semicircle_positive_acceleration: 1.0,
    semicircle_negative_acceleration: 0.3,
    sinusoidal: 0.0,
    exponential: 0.9,
    constant: 0.5,  # f(0) is not 0, makes cliffs
}

def get_elevation_change_function(spikiness=0.5):
    assert 0 <= spikiness <= 1, "spikiness must be in interval [0, 1]"
    choices = []
    weights = []
    for f, f_spikiness in ELEVATION_CHANGE_FUNCTIONS.items():
        choices.append(f)
        weights.append(1 - abs(spikiness - f_spikiness))
    weights = np.array(weights)
    weights /= sum(weights)
    # print("choice", list(zip(choices, weights)))
    return np.random.choice(choices, p=weights)

def show_elevation_change_functions():
    xs = np.linspace(0, 1, 1000)
    max_x = 1
    max_change = 1
    for i, f in enumerate(ELEVATION_CHANGE_FUNCTIONS):
        ys = [f(x, max_x, max_change) for x in xs]
        plt.plot(xs, ys)
        plt.title("function index {}".format(i))
        plt.show()


if __name__ == "__main__":
    show_elevation_change_functions()
