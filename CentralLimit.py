import random
import numpy as np
import matplotlib.pyplot as plt


def sample_and_plot(dist):
    assert callable(dist)
    
    # first plot a histogram of dist values
    xs = [dist() for i in range(10000)]
    plt.subplot(2, 1, 1)
    plt.hist(xs, bins=100, color="blue")
    plt.title("distribution")

    # now get sample means
    n_sample_means = 10000
    n_per_sample = 1000
    sample_means = []
    for i in range(n_sample_means):
        sample_xs = [dist() for j in range(n_per_sample)]
        sample_mean = np.mean(sample_xs)
        sample_means.append(sample_mean)
    plt.subplot(2, 1, 2)
    plt.hist(sample_means, bins=100, color="red")
    plt.title("sample means from same distribution")

    plt.show()


def get_dist():
    typ = random.choice(["pareto", "uniform", "rayleigh", "poisson", "logistic", "zipf"])
    print("chose distribution type: {}".format(typ))
    if typ == "pareto":
        a = random.uniform(1, 3)
        # note that pareto alpha can be >=0, but if alpha <= 1, then the variance is infinite and CLT is no longer valid
        return lambda a=a: np.random.pareto(a)
    if typ == "uniform":
        return lambda: np.random.uniform(0, 1)
    if typ == "rayleigh":
        scale = random.uniform(0, 2)
        return lambda scale=scale: np.random.rayleigh(scale)
    if typ == "poisson":
        lam = random.uniform(0, 10)
        return lambda lam=lam: np.random.poisson(lam)
    if typ == "logistic":
        loc = random.uniform(-10, 10)
        scale = random.uniform(0, 5)
        return lambda loc=loc, scale=scale: np.random.logistic(loc, scale)
    if typ == "zipf":
        a = random.uniform(1, 5)
        return lambda a=a: np.random.zipf(a)
    raise Exception("unknown distribution {}".format(typ))


if __name__ == "__main__":
    dist = get_dist()
    sample_and_plot(dist)
