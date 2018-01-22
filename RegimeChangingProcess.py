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

def get_time_series():
    x, a, b, c = 0, 0, 0, 0
    while True:
        print(x, a, b, c)
        x_s = get_step_normal(a, b)
        a_s = r(-0.01, 0.01) #get_step(b, c, x)
        b_s = r(-0.01, 0.01) #get_step(c, x, a)
        # c_s = get_step_with_restorative_force(c, 0, 0.01) #get_step(x, a, b)
        x += x_s
        a += a_s
        b += b_s
        # c += c_s
        yield x


if __name__ == "__main__":
    xs = [i for i in range(100000)]
    g = get_time_series()
    ys = [next(g) for x in xs]
    plt.plot(xs, ys)
    plt.show()
