# want to plot elliptic curve in box mod a given prime
# curve is: y^2 + y = x^3 - x^2
# see https://oeis.org/A006571
# see https://en.wikipedia.org/wiki/Modularity_theorem
# see something about l-functions of elliptic curves
# youtube lectures by Edward Frenkel about Langlands Program
# for prime p, this curve has n integer solutions mod p (so you only look at ints mod p, and see if equation is satisfied mod p)
# for example, (0, 0) is an easy solution
# for p=5, another solution is (0, 4) because 4^2 + 4 = 20 and 0^3 - 0^2 = 0, which are equal mod 5; there are 4 solutions mod 5
# for prime p, let f(p) be p - n = p minus the number of such solutions mod p
# this creates the series f(2) = -2, f(3) = -1, f(5) = 1
# now look at the expansion of the infinite product:
# q * [ (1-q)^2 * (1-q^2)^2 * (1-q^3)^2 * ... ] * [ (1-q^11)^2 * (1-q^22)^2 * (1-q^33)^2 * ... ]
# = q * prod[k>=1, (1-q^k)^2 ] * prod[k>=1, (1-q^(11*k))^2 ]
# the coefficients on q^p in this polynomial are the same as the values f(p)
# this is a connection between number theory (elliptic curves) and harmonic analysis (modular forms)
# which I don't really understand but it's cool that this connection exists

# try to plot these curves mod p and label their solutions with points

# https://stackoverflow.com/questions/19756043/python-matplotlib-elliptic-curves


import numpy as np
import matplotlib.pyplot as plt


def f(x, y, offset=(0, 0)):
    x_offset, y_offset = offset
    # e.g. if offset is (5, 15), we are actually taking the values of f in the box with lower left corner at (5, 15) instead of (0, 0)
    # so we will get (0+x, 0+y) input but want to return f(5+x, 15+y)
    x = x + x_offset
    y = y + y_offset
    return pow(y, 2) + y - pow(x, 3) + pow(x, 2)


def plot_curve(p):
    # a = -1
    # b = 1

    y, x = np.ogrid[0:p:100j, 0:p:100j]  # passing complex number 100j as step makes it put 100 points in the range, nice shortcut!
    n_boxes = 10
    for xi in range(-n_boxes, n_boxes + 1):
        for yi in range(-n_boxes, n_boxes + 1):
            offset = (xi * p, yi * p)
            plt.contour(x.ravel(), y.ravel(), f(x, y, offset), [0])

    candidates = [(x, y) for x in range(p) for y in range(p)]
    solutions = [pt for pt in candidates if f(*pt) % p == 0]
    for x, y in solutions:
        plt.scatter(x, y, c="r")
    plt.title("p = {}, {} solutions".format(p, len(solutions)))

    # plt.contour(x.ravel(), y.ravel(), pow(y, 2) - pow(x, 3) - x * a - b, [-1])
    # plt.contour(x.ravel(), y.ravel(), pow(y, 2) - pow(x, 3) - x * a - b, [0])
    # plt.contour(x.ravel(), y.ravel(), pow(y, 2) - pow(x, 3) - x * a - b, [1])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    for p in [2, 3, 5, 7, 11]:
        plot_curve(p)
