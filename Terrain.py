import numpy as np
import matplotlib.pyplot as plt


def wave_function():
    a, b, c, d = np.random.uniform(1, 4, (4,))
    return lambda x, y: np.sin(a * x + b) + np.sin(c * y + d)


def get_waves(x_min, x_max, step):
    f = wave_function()
    array = np.arange(x_min, x_max, step)
    return np.array([[f(x, y) for y in array] for x in array])


if __name__ == "__main__":
    x_min = 0
    x_max = 10
    step = 0.01
    n_waves = 5

    waves = get_waves(x_min, x_max, step)
    for i in range(n_waves - 1):
        waves += get_waves(x_min, x_max, step)

    plt.imshow(waves)
    plt.show()