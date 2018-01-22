# x is random walk process
# each step is the sum of a normal variable (for drift and variance) and a s

# mu is random walk in reals
# sigma and alpha are random walks in log space

# can each of these variables' random walks themselves be parametrized in the same way as the output process?
# x can be sigma of mu-generating process, for instance
# so we can have a matrix such as this:
# output   mu sigma alpha
# x[n+1] m[n]  s[n]  a[n]
# m[n+1] s[n]  a[n]  x[n]
# s[n+1] a[n]  x[n]  m[n]
# a[n+1] x[n]  m[n]  s[n]

# but may start with m, s, and a just being simple random walks to simplify things

import numpy as np
import matplotlib.pyplot as plt


def r(a, b):
    r_0_1 = np.random.random()
    return from_0_1_to_interval(r_0_1, a, b)

def from_0_1_to_interval(x, a, b):
    return a + x * (b - a)

def get_step_normal(mu, log_sigma):
    sigma = np.exp(log_sigma)
    return np.random.normal(mu, sigma)

def get_step_normal_plus_pareto(mu, log_sigma, log_alpha):
    # too jumpy!
    sigma = np.exp(log_sigma)
    alpha = np.exp(log_alpha)
    return np.random.normal(mu, sigma) + np.random.choice([-1, 1]) * np.random.pareto(alpha)

def get_step_with_restorative_force(x, center, max_step_size):
    d = x - center
    sigmoid = 1 / (1 + np.exp(-1 * d))  # in [0, 1]
    # area under the sigmoid is probability of moving farther down, above of moving farther up, then within those just uniform
    if np.random.random() < sigmoid:
        return r(-1*max_step_size, 0)
    else:
        return r(0, max_step_size)

def get_count_generator():
    x = 0
    while True:
        yield x
        x += 1

def get_time_series():
    x, a, b, c = 0, 0, 0, 0
    float_format = "{:15.4f}"
    while True:
        # print("".join([float_format.format(item) for item in [x, a, b]]))
        x_s = get_step_normal(a, b)
        # a_s = get_step_with_restorative_force(a, 0, 0.01)
        a_s = r(-0.01, 0.01)
        b_s = r(-0.01, 0.01) #get_step(c, x, a)
        # c_s = get_step_with_restorative_force(c, 0, 0.01) #get_step(x, a, b)
        x += x_s
        a += a_s
        b += b_s
        # c += c_s
        yield (x, a, b, c)

def run_realtime_plot(n_points, plot_window, plot_interval):
    xg = get_count_generator()
    yg = get_time_series()
    plt.ion()
    xs = [0] * plot_window
    ys = [0] * plot_window
    all_xs = []
    all_ys = []
    for i in range(n_points):
        x = next(xg)
        y = next(yg)[0]
        xs.append(x)
        ys.append(y)
        xs = xs[1:]
        ys = ys[1:]
        all_xs.append(x)
        all_ys.append(y)
        if i % plot_interval == 0:
            plt.gcf().clear()
            plt.plot(xs, ys, color="b")
            plt.pause(1e-12)

def run_then_plot(n_points):
    xs = [i for i in range(n_points)]
    yg = get_time_series()
    y_tuples = [next(yg) for x in xs]
    ys = [y[0] for y in y_tuples]
    mus = [y[1] for y in y_tuples]
    log_sigmas = [y[2] for y in y_tuples]
    plt.subplot(211)
    plt.plot(xs, ys, label="y", color="b")
    plt.legend()
    plt.subplot(212)
    plt.plot(xs, log_sigmas, label="log_sigma", color="r")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    n_points = 1000000
    # plot_interval = 100
    # plot_window = 10000
    run_then_plot(n_points)
    # run_realtime_plot(n_points, plot_window, plot_interval)
