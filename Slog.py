import numpy as np
import matplotlib.pyplot as plt
import random


def slog(x):
    # signed log
    return np.sign(x) * np.log(1 + abs(x))


def test_slog():
    xs = np.linspace(-5, 5, 100)
    ys = slog(xs)
    plt.plot(xs, ys)
    plt.show()


def test_slog_power_law_dist():
    # random walk that can change sign and span many orders of magnitude
    # don't just flip sign by multiplication, do some addition terms and sometimes it will cross zero
    xs = [1]
    for i in range(100):
        x = xs[-1]
        if random.random() < 0.5:
            # addition
            y = np.random.pareto(a=1) * random.choice([-1, 1])
            xs.append(x + y)
        else:
            # multiplication
            power = np.random.normal(-10, 10)
            y = 2 ** power
            xs.append(x * y)
    xs = np.array(xs)
    ts = range(len(xs))
    ys = slog(xs)
    plt.plot(ts, ys)
    plt.show()


if __name__ == "__main__":
    # test_slog()
    test_slog_power_law_dist()
