# idea: explain some of a random process's motion by "trend", which here will be a known smooth function,
# and the rest by "swing", meaning motion strictly due to variance around the true underlying trend
# do this for the delta between some pair of points, get the "swing proportion" which is probably just swing/trend
# make 2d graph of this for all pairs of points, like the moment of inertia stick


import random

import numpy as np
import matplotlib.pyplot as plt


def get_trend():
    # just something simple but nonlinear
    a, b, c, d = np.random.uniform(-2, 2, size=(4,))
    f = lambda x: 0.001*a*x + b*np.sin(c*x/50 + d)
    x = 0
    while True:
        yield f(x) - f(0)
        x += 1


def get_noise():
    # EV 0
    x = 0
    while True:
        yield x
        a = np.random.uniform(0.01, 0.1)
        x += a * (2 * random.random() - 1)


class ProcessPoint:
    def __init__(self, trend, noise):
        self.trend = trend
        self.noise = noise
        self.value = trend + noise


class ProcessDelta:
    def __init__(self, p0, p1):
        self.trend_delta = p1.trend - p0.trend
        self.swing = p1.noise - p0.noise
        self.delta = p1.value - p0.value
        math_check = self.delta - (self.trend_delta + self.swing)
        assert abs(math_check) < 1e-6, str(math_check)
        self.total_magnitude = abs(self.trend_delta) + abs(self.swing)
        is_non_negative = lambda x: abs(x) == x
        signs_same = is_non_negative(self.trend_delta) == is_non_negative(self.swing)
        self.swing_proportion = (abs(self.swing) / self.total_magnitude) * (1 if signs_same else -1) if self.total_magnitude != 0 else 0


def get_process():
    trend = get_trend()
    noise = get_noise()
    while True:
        p = ProcessPoint(next(trend), next(noise))
        yield p


if __name__ == "__main__":
    process = get_process()
    points = [next(process) for i in range(1000)]
    # for p in points:
    #     print(p.value, p.noise, "=", p.value, ";")

    plt.subplot(1, 2, 1)

    plt.plot([p.value for p in points], color="k")
    plt.plot([p.trend for p in points], color="b")
    plt.plot([p.noise for p in points], color="r")

    plt.subplot(1, 2, 2)

    grid = [[ProcessDelta(p0, p1).swing_proportion for p1 in points] for p0 in points]
    plt.imshow(grid)
    plt.colorbar()
    plt.show()
