import numpy as np
import matplotlib.pyplot as plt


def get_trajectory_of_differential_equation(dx_funcs, x0s, t_max, dt, max_abs):
    xs = x0s
    x_histories = [[x0] for x0 in x0s]
    n_steps = int(round(t_max / dt))
    for i in range(n_steps):
        new_xs = [None for j in range(len(xs))]
        for j, x in enumerate(xs):
            new_xs[j] = xs[j] + dx_funcs[j](*xs, dt)
            x_histories[j].append(new_xs[j])
        mag = np.linalg.norm(xs)
        if mag > max_abs:
            raise DivergenceError
        xs = new_xs
        print(xs)
    return x_histories


class DivergenceError(Exception):
    pass

