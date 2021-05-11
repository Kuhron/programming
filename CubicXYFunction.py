import numpy as np
import matplotlib.pyplot as plt
import random
import math


class CubicXYFunction:
    def __init__(self, coefficient_arr):
        # array of x powers and y powers from 0 to 3 inclusive
        assert coefficient_arr.shape == (4, 4)
        self.coefficient_arr = coefficient_arr

    @staticmethod
    def random(stdev=1):
        arr = np.random.normal(0, stdev, (4,4))
        return CubicXYFunction(arr)

    def __call__(self, x, y):
        # xs are rows, ys are column
        # WARNING do not use np.outer on arrays of multiple points to multiple powers, it works with a single point but no longer does what you want and explodes in memory for multiple points
        power_grid = np.array([[x**px * y**py for py in range(4)] for px in range(4)])
        term_grid = np.array([[self.coefficient_arr[i,j] * power_grid[i,j] for j in range(4)] for i in range(4)])
        res = term_grid.sum(axis=(0,1))  # sum over first and second axes and be left with point grid shape
        assert res.shape == x.shape == y.shape
        return res

    def plot(self):
        xs = np.linspace(-10, 10, 100)
        ys = np.linspace(-10, 10, 100)
        X,Y = np.meshgrid(xs, ys)
        Z = self(X,Y)
        plt.imshow(Z)
        plt.colorbar()
        plt.contour(Z, levels=[0])
        plt.show()


if __name__ == "__main__":
    f = CubicXYFunction.random()
    print(f.coefficient_arr)
    f.plot()
