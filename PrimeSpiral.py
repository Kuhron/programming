# import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy
import math

# print(sympy.divisor_count(48))

def plot_sacks_spiral(n_points, start_n=1):
    # start_n is at center
    assert start_n >= 1 and n_points >= 1  # i guess you can just plot one point if you feel like it
    max_n = start_n + n_points - 1
    to_effective_n = lambda n: n - start_n
    # archimedes spiral maps first element (effective n = 0) to (0, 0)
    # then effective n = 3 to (1, 0) while spiraling around at constant rate
    # and increasing radius at constant rate
    # r = a*theta
    # r = 0 for theta = 0
    # r = 1 for theta = 2*pi, so a = 1/(2*pi)
    # wolfram alpha: arc length from 0 to e is 1/(4*pi) * (e*sqrt(1+e^2) + sinh^-1(e)), but the inverse of this doesn't have a closed form
    # what if theta is just sqrt(n) or something similar
    # theta(n=0) = 0; theta(n=3) = 2*pi; theta(n=8) = 4*pi;
    # theta(n=x^2-1) = 2*pi*(x-1); x = sqrt(n+1)
    # theta(n) = 2*pi*(sqrt(n+1)-1); try that
    to_theta = lambda n: 2*math.pi*(math.sqrt(n+1)-1)
    to_r = lambda theta: 1/(2*math.pi)*theta
    approx = lambda x, y: abs(x-y) < 1e-9
    assert approx(to_theta(4-1), 2*math.pi)
    assert approx(to_theta(9-1), 4*math.pi)
    xs = []
    ys = []
    sizes = []
    colors = []
    cmap = matplotlib.cm.get_cmap("YlOrRd")
    for nn in range(start_n, max_n + 1):
        n = to_effective_n(nn)
        theta = to_theta(n)
        r = to_r(theta)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        divs = sympy.divisor_count(nn)  # original number, the label, not the position
        divs = int(divs)  # f***ing sympy Integer type!
        xs.append(x)
        ys.append(y)
        sizes.append(divs)

    max_size = max(sizes)
    max_display_size = 12
    colors = [cmap(size/max_size) for size in sizes]
    sizes = [size * max_display_size/max_size for size in sizes]
    plt.scatter(xs, ys, c=colors, s=sizes)
    plt.show()


if __name__ == "__main__":
    plot_sacks_spiral(10000, start_n=1048576)
