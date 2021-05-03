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
            a = np.random.uniform(0, stdev)
            if -1 <= a <= 1:
                break
            i += 1
            if i > 10000:
                raise RuntimeError(f"stdev {stdev} led to too many iterations when trying to roll scale parameter; please reduce stdev")
        n = random.randint(1, 10)
        return DistortionFunction01(frequency=n, amplitude=a)


class DistortionFunctionSeries:
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
        return DistortionFunctionSeries(funcs)


class Mouth:
    def __init__(self, f1_distortion_func, f2_distortion_func):
        self.f1_distortion_func = f1_distortion_func
        self.f2_distortion_func = f2_distortion_func

    @staticmethod
    def random(stdev):
        f1_distortion_func = DistortionFunctionSeries.random(stdev)
        f2_distortion_func = DistortionFunctionSeries.random(stdev)
        return Mouth(f1_distortion_func, f2_distortion_func)

    def convert_articulation_to_formants(self, arts):
        f1_x, f2_x = arts
        f1 = self.distort_f1(f1_x)
        f2 = self.distort_f2(f2_x)
        return f1, f2

    def distort_f1(self, x):
        return self.f1_distortion_func(x)

    def distort_f2(self, x):
        return self.f2_distortion_func(x)

    def plot_distortions(self):
        xs = np.linspace(0,1,101)
        f1s = self.distort_f1(xs)
        f2s = self.distort_f2(xs)
        plt.plot(xs, f1s, label="f1", c="r")
        plt.plot(xs, f2s, label="f2", c="b")
        plt.plot(xs, xs, label="y=x", c="#777777")
        plt.legend()
        plt.show()


def homebrew_interpolate(x, xs, ys):
    below_xys = [(a,b) for a,b in zip(xs, ys) if a <= x]
    above_xys = [(a,b) for a,b in zip(xs, ys) if a >= x]
    x0, y0 = below_xys[-1]
    x1, y1 = above_xys[0]
    # assumes xs are sorted
    if x0 == x1:
        assert y0 == y1
        return y0
    return y0 + (y1-y0) * (x-x0)/(x1-x0)


if __name__ == "__main__":
    stdev = 0.5  # inter-speaker anatomical variation
    m1 = Mouth.random(stdev)
    m2 = Mouth.random(stdev)
    m3 = Mouth.random(stdev)
    m1.plot_distortions()

    articulation = np.random.uniform(0, 1, (2,))
    fs1 = m1.convert_articulation_to_formants(articulation)
    fs2 = m2.convert_articulation_to_formants(articulation)
    fs3 = m3.convert_articulation_to_formants(articulation)
    plt.scatter(*articulation, marker="x")
    plt.scatter(*fs1)
    plt.scatter(*fs2)
    plt.scatter(*fs3)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
