# inspired by trying to find patterns in this post:
# - https://www.facebook.com/photo/?fbid=10168861055085627&set=pcb.1855714428185427
# - downloaded image as 449452197_10168861055090627_3420005998891624849_n.jpg


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math
from sympy.ntheory import factorint
import itertools


# def get_coprimes_less_than_n(n):
#     res = [1]
#     factors_of_n = factorint(n)
#     for x in range(2, n):
#         factors_of_x = factorint(x)
#         if factor_dicts_are_coprime(factors_of_n, factors_of_x):
#             res.append(x)
#     return res


# def factor_dicts_are_coprime(ds):
#     if len(ds) == 0:
#         raise ValueError("need at least one factor dict")
#     s = set(ds[0].keys())
#     for d in ds[1:]:
#         s &= set(d.keys())
#     return len(s) == 0


# def get_rationals(n):
#     # numbers in [0,1] that have n as denominator in reduced form
#     return [0] + [x/n for x in get_coprimes_less_than_n(n)] + [1]


# def get_rational_lattice_points(n):
#     xs = get_rationals(n)
#     # for n > 1, need points where at least one of the coordinates is not 0 or 1
#     if n == 1:
#         return [(0,0), (0,1), (1,0), (1,1)]
#     else:
#         return [(x,y) for x in xs for y in xs if not (x in [0,1] and y in [0,1])]


def get_xys(n):
    xs = list(range(0, n+1))
    return itertools.product(xs, xs)


def point_has_been_seen_before(x, y, n):
    return math.gcd(x,y,n) != 1


def get_color(n, max_n):
    return get_color_red_decay_to_yellow(n, max_n)
    # return get_color_cyclical(n)


def get_color_cyclical(n):
    # hue = get_hue_constant_period(n)
    hue = get_hue_slowing_period(n)

    hsv = (hue, 1, 1)
    return hsv_to_rgb(hsv)


def get_hue_slowing_period(n):
    y = math.sqrt(n-1) % 1
    return y


def get_hue_constant_period(n):
    period = 12
    hue = ((n-1) * 1 / period) % 1
    return hue


def get_color_red_decay_to_yellow(n, max_n):
    RED = np.array([1, 0, 0])
    YELLOW = np.array([1, 1, 0])
    color_half_life = 4

    red_proportion = 2 ** ((1-n) / color_half_life)
    yellow_proportion = 1 - red_proportion
    color = red_proportion * RED + yellow_proportion * YELLOW

    max_n_for_alpha_1 = 12
    # want alpha close to 1 for a while then decay to 0 right before max_n
    if n <= max_n_for_alpha_1:
        alpha = 1
    else:
        # sigmoid in 01 box: https://www.desmos.com/calculator/wwyno6g9xh
        proportion_along_decline = (n - max_n_for_alpha_1) / (max_n+1 - max_n_for_alpha_1)
        x = proportion_along_decline
        a = -3
        c = 0.5
        b = math.log(1/c - 1)
        g = a * math.log(1/x - 1) + b
        alpha = 1 / (1 + math.exp(g))
    # don't use alpha as actual opacity, use it as decay to black
    color *= alpha

    return color



if __name__ == "__main__":
    max_n = 200
    side_length_inches = 32
    max_size = 2000 * side_length_inches / 16

    xs = []
    ys = []
    colors = []
    sizes = []

    for n in range(max_n, 0, -1):
        # draw the small points first so the big ones are on top of them
        color = get_color(n, max_n)
        for x,y in get_xys(n):
            # print(f"{(x,y,n)} : {'red' if point_has_been_seen_before(x,y,n) else 'yellow'}")
            if not point_has_been_seen_before(x,y,n):
                xs.append(x/n)
                ys.append(y/n)

                colors.append(color)

                size = max_size / (n**2)
                sizes.append(size)

    fig, ax = plt.subplots()
    fig.set_size_inches(side_length_inches, side_length_inches)
    plt.scatter(xs, ys, c=colors, s=sizes)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.gcf().set_facecolor("black")
    plt.gca().set_facecolor("black")
    plt.savefig("RationalGrid.png")

