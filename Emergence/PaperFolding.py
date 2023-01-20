# idea: have a function that maps the [0,1]*[0,1] square to itself, by:
# 1. make some continuous mapping that stretches it into a blob or whatever, just do any function on it to R^2
# 2. "fold the paper" that reaches outside the box back into the box. Do this by going in some ordering on the four edges of the box, e.g. right, bottom, left, top, and take everything that lies farther away than that edge and fold it over the edge so that it is closer to the box and some of it will now be in the box. E.g. take the part of the shape that lies at x > 1, fold it by mapping its points (x,y) to (2-x, y), then take the part of the shape that lies at y < 0, fold it by mapping its points (x,y) to (x, -y), left edge: (x,y) -> (-x, y), top edge: (x,y) -> (x, 2-y)
# repeat this folding until the whole shape is back in the box
# that completes one iteration of the function
# then do the whole thing again by applying the continuous map then doing the paper-folding, iterate
# see what attractors/fractals emerge from this
# can see how ordering the edge foldings differently changes the outcome
# can even give it non-periodic folding orders, like primes mod 4 or something like that to select which edge is next


import numpy as np
import matplotlib.pyplot as plt
import math



def fold_point_over_right_edge(x, y):
    x = np.where(x > 1, 2-x, x)
    return x, y


def fold_point_over_left_edge(x, y):
    x = np.where(x < 0, -x, x)
    return x, y


def fold_point_over_top_edge(x, y):
    y = np.where(y > 1, 2-y, y)
    return x, y


def fold_point_over_bottom_edge(x, y):
    y = np.where(y < 0, -y, y)
    return x, y


def get_edge_order_repeating(seq):
    while True:
        for x in seq:
            yield x


def get_edge_order_default():
    return get_edge_order_repeating(["R", "T", "L", "B"])


def fold_point_over_edge_by_letter(x, y, edge):
    if edge == "R":
        return fold_point_over_right_edge(x, y)
    elif edge == "L":
        return fold_point_over_left_edge(x, y)
    elif edge == "T":
        return fold_point_over_top_edge(x, y)
    elif edge == "B":
        return fold_point_over_bottom_edge(x, y)
    else:
        raise ValueError(f"unknown edge {edge}")


def is_in_box(x, y):
    return (0 <= x).all() and (x <= 1).all() and (0 <= y).all() and (y <= 1).all()


def fold(x, y, edges):
    # fold it over each edge in the ordering until it is back inside the box
    for edge in edges:
        # check first if we even need to fold at all, maybe it's already fine
        if is_in_box(x, y):
            return x, y
        x, y = fold_point_over_edge_by_letter(x, y, edge)
    raise Exception("shouldn't get here")


def apply_function_with_fold(f, x, y, edges):
    x, y = f(x, y)
    x, y = fold(x, y, edges)
    return x, y


def apply_function_with_mod(f, x, y):
    x, y = f(x, y)
    x = x % 1
    y = y % 1
    return x, y



if __name__ == "__main__":
    # f0 = lambda x, y: (2*x*y**2 - y*x**2 - 6, -x**2 + 1/3 * y**2*x + 3)  # original function
    # a = [-6, 0, 0, 0, 0, 0, 0, -1, 2, 0, 3, 0, 0, -1, 0, 0, 0, 0, 1/3, 0]  # array for original function (degree 3)
    mask = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 0] * 2)  # trying to see if something about the terms I included in the original function is why I got such a good shape
    a = np.random.randint(-5, 6, (20,))
    a[np.random.random(20) < 0.5] = 0
    a[mask == 0] = 0
    print(list(a))  # list() so it will print commas for easier pasting into Python
    f0 = lambda x, y: (
        + a[0]
        + a[1] * x
        + a[2] * y
        + a[3] * x**2
        + a[4] * x * y
        + a[5] * y**2
        + a[6] * x**3
        + a[7] * x**2 * y
        + a[8] * x * y**2
        + a[9] * y**3,

        + a[10]
        + a[11] * x
        + a[12] * y
        + a[13] * x**2
        + a[14] * x * y
        + a[15] * y**2
        + a[16] * x**3
        + a[17] * x**2 * y
        + a[18] * x * y**2
        + a[19] * y**3,
    )

    resolution = 100
    xs = np.linspace(0, 1, resolution)
    ys = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(xs, ys)

    n_steps = 50
    for i in range(n_steps):
        edges = get_edge_order_default()  # remake the generator each time so it starts in the same place
        X1, Y1 = apply_function_with_fold(f0, X, Y, edges)
        dx = X1 - X
        dy = Y1 - Y
        X, Y = X1, Y1
        if abs(dx).max() < 1e-6 or abs(dy).max() < 1e-6:
            break
    fold_attractor = X, Y

    X, Y = np.meshgrid(xs, ys)
    for i in range(n_steps):
        X, Y = apply_function_with_mod(f0, X, Y)
    mod_attractor = X, Y

    plt.subplot(1,2,1)
    plt.scatter(*fold_attractor, alpha=0.5, c="k", s=4, edgecolors="none")
    plt.title("fold")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.subplot(1,2,2)
    plt.scatter(*mod_attractor, alpha=0.5, c="k", s=4, edgecolors="none")
    plt.title("mod")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()


    # discoveries
    # [-2,-1, 0, 0, 0, 0, 0, 0,-5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # good mod
    # [-3, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 4, 0,-5, 0, 0, 0, 0, 0]  # good mod
    # [ 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,-3, 0, 0, 0, 0]  # both really interesting, mod has a bell curve shape
    # [ 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 1, 0, 0, 0,-1, 0, 0,-1, 0]  # good mod
    # [ 0,-5, 0, 0, 0, 0, 0, 1, 0,-5,-4, 0, 0, 0, 0, 0, 5, 0, 0, 0]  # clear relationship between fold and mod
    # [ 2, 0, 0,-4, 0, 0, 0, 0, 4, 0,-1, 0, 0, 0, 0, 0, 0, 0,-2, 0]  # good fold (curves)
    # [-4, 0, 0, 0, 0, 1, 0, 0, 0, 0,-3, 0, 0, 0, 0, 1, 0, 3, 0, 0]  # good fold (ring)

