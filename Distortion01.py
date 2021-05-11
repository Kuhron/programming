import numpy as np
import math
import random
import matplotlib.pyplot as plt


class DistortionFunction01:
    def __init__(self, frequency, amplitude):
        assert frequency > 0
        assert frequency % 1 == 0
        assert -1 <= amplitude <= 1
        self.frequency = frequency
        self.amplitude = amplitude

    def __call__(self, x):
        a = self.amplitude
        n = self.frequency
        n_pi = n * np.pi
        return x + a * 1/n_pi * np.sin(n_pi * x)
        # maps [0,1] interval to itself with some bending, still monotonic
        # so iterating different functions of this form can give you wiggly shape that is still 1-to-1 in [0,1] and monotonically increasing

    @staticmethod
    def random(stdev):
        i = 0
        while True:
            a = np.random.normal(0, stdev)
            if -1 <= a <= 1:
                break
            i += 1
            if i > 10000:
                raise RuntimeError(f"stdev {stdev} led to too many iterations when trying to roll scale parameter; please reduce stdev")
        n = random.randint(1, 10)
        return DistortionFunction01(frequency=n, amplitude=a)

    @staticmethod
    def regression_to_mean(stdev):
        # a should be between 0 and 1 so that more things map toward the middle and away from the edges
        while True:
            a = abs(np.random.normal(0, stdev))
            if 0 <= a <= 1:
                break
        n = 2  # only one period inside the box
        return DistortionFunction01(frequency=n, amplitude=a)

    def plot(self):
        xs = np.linspace(0, 1, 101)
        ys = self(xs)
        plt.plot(xs, ys)
        plt.show()

    def plot_image_distribution(self):
        # image as in the image of the function
        xs = np.linspace(0, 1, 10001)
        ys = self(xs)
        plt.hist(ys, bins=100)
        plt.show()


class DistortionFunctionSeries01:
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, x):
        for f in self.funcs:
            x = f(x)
        return x

    @staticmethod
    def random(stdev):
        n_funcs = random.randint(3, 8)
        funcs = [DistortionFunction01.random(stdev) for i in range(n_funcs)]
        return DistortionFunctionSeries01(funcs)

