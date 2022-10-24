# map square [0,1]^2 onto itself with shear and cut function

import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.insert(0, "/home/wesley/programming/")
from PltContentOnly import imshow_content_only


get_signed_pareto = lambda: random.choice([-1, 1]) * (np.random.pareto(1))
get_normal = lambda: np.random.normal()


def get_polynomial_func(degree, mod=True):
    coeffs = [get_signed_pareto() for i in range(degree+1)]
    print(coeffs)
    f = lambda x, coeffs=coeffs: sum(coeffs[i] * x**i for i in range(degree+1))
    if mod:
        return lambda x: f(x) % 1
    else:
        return f


def get_bivariate_polynomial_func(degree):
    fxx = get_polynomial_func(degree, mod=False)
    fxy = get_polynomial_func(degree, mod=False)
    fyx = get_polynomial_func(degree, mod=False)
    fyy = get_polynomial_func(degree, mod=False)
    f = lambda x, y, fxx=fxx, fxy=fxy, fyx=fyx, fyy=fyy: ((fxx(x) + fxy(y)) % 1, (fyx(x) + fyy(y)) % 1)
    return np.vectorize(f)


def run():
    f = get_bivariate_polynomial_func(2)

    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xs, ys)
    X = X.flatten()
    Y = Y.flatten()

    heatmap_resolution = 250
    bin_counts_2d = np.zeros((heatmap_resolution, heatmap_resolution))

    n_steps = 1000
    for i in range(n_steps):
        if i % 10 == 0:
            print(f"step {i}/{n_steps}")
        X, Y = f(X, Y)
        for x, y in zip(X, Y):
            # find what bin this point is in
            x_bin_i = int(x / (1 / heatmap_resolution))
            y_bin_i = int(y / (1 / heatmap_resolution))
            bin_counts_2d[x_bin_i, y_bin_i] += 1

    now_str = datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S")
    save_fp = f"Images/ShearMap/{now_str}.png"
    imshow_content_only(bin_counts_2d.T, size_inches=(10,10), save_fp=save_fp, origin="lower")
    plt.gcf().clear()


if __name__ == "__main__":
    while True:
        run()
