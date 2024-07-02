# from Brilliant course on Number Theory, Lesson "100 Doors"
# plot how the doors change between open and closed at each step
# and can use a similar approach with different rules for how the doors change, to see what the plots look like

import numpy as np
import matplotlib.pyplot as plt
import math


def perform_step(vec, step_i):
    n_doors = len(vec)
    # flip each door at multiples of step_i, 1-indexed
    max_m = n_doors // step_i
    for m in range(1, max_m+1):
        x = step_i * m
        vec[x-1] = 1 - vec[x-1]
    return vec


if __name__ == "__main__":
    n_doors = 2000
    n_steps = 2000
    arr = np.zeros((n_steps + 1, n_doors))
    vec = np.zeros((n_doors,))
    arr[0, :] = [x for x in vec]

    for step_i in range(1, n_steps+1):
        vec = perform_step(vec, step_i)
        arr[step_i, :] = [x for x in vec]

    n_rows, n_cols = arr.shape
    # range (n_cols+1) so we have one extra column point (simil rows) due to pcolormesh needing the corners of the grid (one extra fencepost on each axis)
    # then add 1 to the whole xs and ys arrays so the points are 1-indexed for log scale
    xs = np.arange(n_cols+1) + 1
    ys = np.arange(n_rows+1) + 1
    ys = np.log(ys)
    X, Y = np.meshgrid(xs, ys)

    # shift the columns vertically by a different amount for each x
    log_xs = np.log(xs)
    for x_i in range(len(xs)):
        Y[:, x_i] -= log_xs[x_i]/2

    print(X.shape)
    print(Y.shape)
    print(arr.shape)

    plt.pcolormesh(X, Y, arr)
    plt.show()

