import numpy as np
import matplotlib.pyplot as plt


def apply(f, dist_f, n_samples=10000):
    dist_samples = dist_f(size=n_samples)
    transformed = f(dist_samples)
    return transformed


if __name__ == "__main__":
    a = np.random.uniform(-10, 10)
    b = np.random.uniform(0, 10)
    # dist_f = lambda size: np.random.normal(a, b, size=size)
    dist_f = lambda size: np.random.pareto(a=1, size=size)

    # f = lambda x: abs(x)**0.5 - abs(x)**0.4
    # f = lambda x: x**-x
    f = lambda x: np.log(x)

    xs = apply(f, dist_f, n_samples=1000000)
    plt.hist(xs, bins=100)
    plt.show()
