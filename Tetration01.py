# what do tetration functions look like on [0, 1]?
# 0 ^^ n alternates between 1 and 0 (0 for odd n)
# 1 ^^ n is always 1
# for odd n, the curve lifts away from y=x line
# for even n, the curve falls away from x^x curve
# these alternate and seem to converge on some curve in the middle
# what is the limiting curve?
# and what is the limit as x->0 and n->inf of x ^^ n?
# i.e., where does the limiting curve intersect the y-axis?

# ok so it doesn't actually touch the y-axis, it converges to a bifurcation
# what are the equations of these three curves in the middle?
# (the limit curves that lie inside the region bounded by y=x, y=x^x, and x=0)
# see https://en.wikipedia.org/wiki/File%3aInfinite_power_tower.svg
# the lower limit is e^-e, and it converges until x=e^(1/e)
# and according to this: https://en.wikipedia.org/wiki/Tetration#Infinite_heights
# the resulting x is a solution of y = x^y, which requires use of the Lambert W function to express


import numpy as np
import matplotlib.pyplot as plt

def tetrate(x, n):
    assert n >= 1
    assert n % 1 == 0
    assert 0 <= x <= 1  # not necessary for tetration, but don't want to deal with other stuff right now

    result = x
    for _ in range(n - 1):  # if n == 1, just return x, so don't enter loop
        result = x ** result
    return result

if __name__ == "__main__":
    x_min = 0.06598
    x_max = 0.06600
    n_pts = 100
    step = (x_max - x_min) / n_pts
    xs = np.arange(x_min, x_max + step, step)
    for n in [100000, 100001, 1000000, 1000001]:
        ys = [tetrate(x, n) for x in xs]
        plt.plot(xs, ys)
    plt.show()


