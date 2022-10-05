# from https://www.youtube.com/watch?v=R9FKN9MIHlE&t=16s

import random
import numpy as np
import matplotlib.pyplot as plt


def get_primes(max_p):
    ns = list(range(2, max_p+1))
    res = []
    while len(ns) > 0:
        p = ns[0]
        mults = [i*p for i in range(1, max_p//p + 1)]
        res.append(p)
        ns = [x for x in ns if x not in mults]
    return res


def evaluate_elliptic_curve(x, y, a, b):
    return y**2 - x**3 - a*x - b


def evaluate_elliptic_curve_mod_p(x, y, a, b, p):
    return evaluate_elliptic_curve(x, y, a, b) % p


def get_solutions_mod_p(a, b, p):
    xs = []
    ys = []
    for x in range(0, p + 1):
        for y in range(0, p + 1):
            z = evaluate_elliptic_curve_mod_p(x, y, a, b, p)
            if z == 0:
                print(f"integer solution found: ({x}, {y}) mod {p}")
                xs.append(x)
                ys.append(y)
    return xs, ys


if __name__ == "__main__":
    a = random.randint(-10, 10)
    b = random.randint(-10, 10)
    primes = get_primes(100)
    for p in primes:
        xs, ys = get_solutions_mod_p(a, b, p)
        plt.scatter(xs, ys)
        plt.title(f"a={a}, b={b}, p={p}")
        plt.savefig(f"Images/EllipticCurves/a{a}_b{b}_p{p}.png")
        plt.gcf().clear()

