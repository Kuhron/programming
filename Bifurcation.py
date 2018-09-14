import numpy as np
import matplotlib.pyplot as plt


def logistic_map(x, r):
    return r * x * (1 - x)


def iterate(f, x0, *args, **kwargs):
    x = x0
    while True:
        yield x
        x = f(x, *args, **kwargs)


def get_iterations(f, x0, n0, n1, *args, **kwargs):
    # get n0-th through n1-th iterations
    g = iterate(f, x0, *args, **kwargs)
    for _ in range(n0):
        next(g)  # throw it away
    return [next(g) for _ in range(n1 - n0)]


def find_equilibria(f, x0, *args, **kwargs):
    # how to do this, finding where it converges?
    # could specify some large N, find the range of the orbit between, say, N and N+1024 steps. Plot all those points.
    N = 10000
    n = 1024
    iterations = get_iterations(f, x0, N, N+n, *args, **kwargs)
    return sorted(set(iterations))


def scatter(f, x0, n0, n1, *args, **kwargs):
    ys = get_iterations(f, x0, n0, n1, *args, **kwargs)
    plt.scatter(range(n0, n1), ys)
    plt.show()


# def plot_equilibria(f, x0, *args, **kwargs):
#     plt.


def plot_bifurcation_diagram(f, x0, arg_min, arg_max):
    # f should be func of one variable, plotting only bifurcation along that one and holding all else constant
    fail_str = "f must be function of one variable for bifurcation plot, returning a func of x; e.g. f = lambda r: lambda x: r * x * (1 - x)"
    assert f.__code__.co_argcount == 1, fail_str
    first_func_to_iterate = f(arg_min)  # pass the parameter to get the function of x
    assert callable(first_func_to_iterate), fail_str
    try:
        first_func_to_iterate(x0)  # checks that it works when passed the one arg
    except:
        raise Exception(fail_str)

    n_points = 1000
    arg_range = arg_max - arg_min
    arg_lst = np.arange(arg_min, arg_max, arg_range/n_points)

    xs = []
    ys = []

    for arg in arg_lst:
        f_arg = f(arg)
        equilibria = find_equilibria(f_arg, x0)
        # print(arg, equilibria)
        for eq in equilibria:
            xs.append(arg)
            ys.append(eq)

    plt.scatter(xs, ys, c="k", marker=",")
    plt.show()


if __name__ == "__main__":
    # scatter((lambda x: (2*x+1) % 100), 0, 10000, 11024)  # test
    # scatter(logistic_map, 0.01, 10000, 11024, 3.5)

    plot_bifurcation_diagram((lambda r: lambda x: logistic_map(x, r)), 0.5, 2.5, 4)



