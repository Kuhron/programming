import random
import time
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

from InteractivePlot import InteractivePlot


# make a function in 2D, show it with contours
# ideally toroidal array and the function wraps around continuously
# start at a random point and apply various algorithms for finding the global minimum
# e.g. gradient descent stuff, momentum, etc., other stuff that ML/NNs use, to visualize what the optimizer algorithms do


def get_random_point_in_box():
    return np.random.uniform(0, 2*pi, (2,))


def get_random_function_and_gradient():
    def get_funcs():
        # xs = np.linspace(0, 2*pi, 100)
        # ys = np.vectorize( lambda x: (a * np.sin(n*x + b)).sum() )( xs )

        spectrum = lambda n: 1.25 ** -np.maximum(2,n)

        n = 8
        N = np.arange(1, 1+n)
        Ax = np.random.uniform(-1, 1, n) * spectrum(N)
        Ay = np.random.uniform(-1, 1, n) * spectrum(N)
        Bx = np.random.uniform(0, 2*pi, n)
        By = np.random.uniform(0, 2*pi, n)

        F1x = np.vectorize(lambda x: (Ax * sin(N*x + Bx)).sum())
        F1y = np.vectorize(lambda y: (Ay * sin(N*y + By)).sum())
        D1x = np.vectorize(lambda x: (Ax * cos(N*x + Bx) * N).sum())
        D1y = np.vectorize(lambda y: (Ay * cos(N*y + By) * N).sum())

        func = lambda x, y: F1x(x) + F1y(y)
        grad = lambda x, y: np.array([D1x(x), D1y(y)])

        return func, grad

    f, grad = get_funcs()

    # check periodicity
    eps = 1e-6
    for i in range(10):
        p = get_random_point_in_box()
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                p2 = p + np.array([dx, dy])*2*pi
                assert abs(f(*p) - f(*p2)) < eps, f"f({p}) = {f(*p)}\nf({p2}) = {f(*p2)}"
                assert np.linalg.norm(grad(*p) - grad(*p2)) < eps, f"grad({p}) = {grad(*p)}\ngrad({p2}) = {grad(*p2)}"
                assert p.shape == grad(*p).shape, f"p = {p}\ngrad({p}) = {grad(*p)}"

    return f, grad


def perturb_vector_slight_rotation(v):
    re, im = v
    z = re + im*1j
    d_theta = np.random.normal(0, np.pi/100)
    z2 = cos(d_theta) + 1j*sin(d_theta)
    z *= z2  # rotate slightly
    return np.array([z.real, z.imag])


def gradient_descent(x, f, grad, learning_rate):
    dx = -1 * learning_rate * grad(*x)
    return x + dx, None


def gradient_descent_with_random_kicks(x, f, grad, learning_rate, kick_rate):
    x2 = gradient_descent(x, f, grad, learning_rate)
    g1 = grad(*x)
    g2 = grad(*x2)
    g3 = grad(*((x + x2) / 2))
    g = min(np.linalg.norm(g1), np.linalg.norm(g2), np.linalg.norm(g3))
    # if the two gradients (at x and x2) cancel out mostly, then we are likely to oscillate
    # the smaller the gradient is here, the more likely we are to need a kick to go somewhere else (to avoid local minimum)
    kick_mag = kick_rate * np.exp(-g)
    kick_direction = np.random.uniform(0, 2*pi)
    kick = kick_mag * np.array([cos(kick_direction), sin(kick_direction)])
    return x2 + kick, None
    # would be nice if it had memory of when it had a good score so it can resist local minima that aren't as good as its record so far


def gradient_descent_with_ema(x, f, grad, learning_rate, ema_gamma, memory):
    g = grad(*x)
    # perturb the gradient's direction slightly
    g = perturb_vector_slight_rotation(g)

    past_ema = memory.get("ema")
    if past_ema is None:
        new_ema = g
    else:
        new_ema = ema_gamma * past_ema + (1 - ema_gamma) * g
    memory["ema"] = new_ema
    ema_grad_magnitude = np.linalg.norm(new_ema)
    # if the ema has magnitude near zero, it indicates we may be stuck in a minimum and/or oscillating, so jack up the learning rate

    learning_rate_adjustment = 1 / ema_grad_magnitude
    effective_learning_rate = learning_rate * learning_rate_adjustment
    dx = -1 * effective_learning_rate * g
    new_x = x + dx
    # TODO oscillation detection still not really working; it correctly jumps when we are sitting at a minimum, but oscillation never gets treated the same way as this
    # TODO the ema-caused jump is too often in the same direction as previous jumps
    return new_x, memory


def get_next_guess(x, f, grad, memory=None):
    # memory is a dictionary used to store stuff for better optimization algorithms, e.g. adaptive things that remember something like EMA of the past gradients
    if memory is None:
        memory = {}

    # return gradient_descent(x, f, grad, learning_rate=0.1)
    # return gradient_descent_with_random_kicks(x, f, grad, learning_rate=0.1, kick_rate=0.5)
    return gradient_descent_with_ema(x, f, grad, learning_rate=0.01, ema_gamma=0.7, memory=memory)


def plot_background(X, Y, Z):
    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.contour(X, Y, Z)


def plot_point_on_background(p, X, Y, Z):
    plot_background(X, Y, Z)
    art1, = plt.plot(*p)
    return [art1]


def plot_point_change(p, new_p):
    x, y = p
    x2, y2 = new_p
    dx = x2 - x
    dy = y2 - y
    art1, = plt.plot([x, x2], [y, y2], c="k")
    art2, = plt.plot(x, y, "o", c="b", markeredgewidth=1, markeredgecolor="k")
    art3, = plt.plot(x2, y2, "o", c="r", markeredgewidth=1, markeredgecolor="k")
    return [art1, art2, art3]


def plot_point_change_on_background(p, new_p, X, Y, Z):
    plot_background(X, Y, Z)
    art_objs = plot_point_change(p, new_p)
    return art_objs



if __name__ == "__main__":
    f, grad = get_random_function_and_gradient()
    xs = np.linspace(0, 2*pi, 200)
    ys = np.linspace(0, 2*pi, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    global_max = Z.max()
    global_min = Z.min()
    transform_score = lambda x: (x - global_min) / (global_max - global_min)

    p = get_random_point_in_box()

    plt.ion()
    fignum = plt.gcf().number  # TODO make this work with InteractivePlot context manager (it works differently in this program because of the functions like contourf rather than just plot, and also because of persisting the background image)
    plt.show()

    art_objs = plot_point_on_background(p, X, Y, Z)
    plt.draw()
    plt.pause(0.01)

    scores = [transform_score(f(*p))]
    memory = None
    while True:
        new_p, memory = get_next_guess(p, f, grad, memory)
        new_p = new_p % (2*pi)  # ideally can show the arrow "wrap around" the torus so it's clearer that an algorithm is overshooting
        # print(f"{p} -> {new_p} (diff {new_p - p})")
        scores.append(transform_score(f(*new_p)))

        for art in art_objs:
            # remove the point markers and lines from last step, but keep the function contour background
            art.remove()
        art_objs = plot_point_change(p, new_p)
        plt.draw()
        plt.pause(0.01)
        if not plt.fignum_exists(fignum):
            print("user closed plot; exiting")
            break

        p = new_p
        # time.sleep(0.01)

    plt.ioff()

    plt.plot(scores)
    plt.title("loss over time")
    plt.show()
