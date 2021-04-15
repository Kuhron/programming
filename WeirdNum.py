import math
import random
import numpy as np
import matplotlib.pyplot as plt


class WeirdNum:
    def __init__(self, n):
        assert type(n) in [int, float, np.float64], "can't make WeirdNum from {}: {}".format(type(n), n)
        self.n = n

    def __add__(self, other):
        x = max(self.n, other.n)
        return WeirdNum(x)

    def __sub__(self, other):
        x = min(self.n, other.n)
        return WeirdNum(x)

    def __mul__(self, other):
        # x = self.n * (self.n + other.n)  # original WeirdNum
        x = self.n + other.n  # tropical geometry
        return WeirdNum(x)

    def __truediv__(self, other):
        x = self.n / (self.n + other.n)
        return WeirdNum(x)

    def __floordiv__(self, other):
        denom = self.n + other.n
        if denom != 0:
            x = (self.n * other.n) // denom
        else:
            x = 0
        return WeirdNum(x)

    def __mod__(self, other):
        denom = self.n + other.n
        if denom != 0:
            x = (self.n * other.n) % denom
        else:
            x = 0
        return WeirdNum(x)

    def __pow__(self, other):
        x = abs(self.n ** other.n - other.n ** self.n)
        return WeirdNum(x)

    def __and__(self, other):
        x = round(self.n) & round(self.n + other.n)
        return WeirdNum(x)

    def __or__(self, other):
        x = round(self.n) | round(self.n + other.n)
        return WeirdNum(x)

    def __xor__(self, other):
        x = round(self.n) ^ round(self.n + other.n)
        return WeirdNum(x)

    def __invert__(self):  # this is the ~ operator
        x = 1/self.n if self.n != 0 else 0
        return WeirdNum(x)

    def __neg__(self):  # unary -
        x = -1/self.n if self.n != 0 else 0
        return WeirdNum(x)

    def __gt__(self, other):
        a = (self.n + other.n) % self.n if self.n != 0 else 0
        b = (self.n + other.n) % other.n if other.n != 0 else 0
        x = a > b
        return x

    def __eq__(self, other):
        a = (self.n + other.n) % self.n if self.n != 0 else 0
        b = (self.n + other.n) % other.n if other.n != 0 else 0
        x = a == b
        return x

    def __repr__(self):
        return "{}'".format(self.n)


def report_operations_on_random_pair():
    a = random.randint(0, 10)
    b = random.randint(0, 10)
    a = WeirdNum(a)
    b = WeirdNum(b)
    z = WeirdNum(0)
    print("{} + {} = {}".format(a, b, a+b))
    print("{} - {} = {}".format(a, b, a-b))
    print("{} * {} = {}".format(a, b, a*b))
    print("{} / {} = {}".format(a, b, a/b))
    print("{} // {} = {}".format(a, b, a//b))
    print("{} % {} = {}".format(a, b, a%b))
    print("{} ** {} = {}".format(a, b, a**b))
    print("{} & {} = {}".format(a, b, a&b))
    print("{} | {} = {}".format(a, b, a|b))
    print("{} ^ {} = {}".format(a, b, a^b))
    print("~{} = {}".format(a, ~a))
    print("~{} = {}".format(b, ~b))
    print("{} < {} = {}".format(a, b, a<b))
    print("{} == {} = {}".format(a, b, a==b))
    print("-{} = {}".format(a, -a))
    print("-{} = {}".format(b, -b))
    print("{} - {} = {}".format(z, a, z-a))
    print("{} - {} = {}".format(z, b, z-b))


def plot_function(f):
    xmin = -40
    xmax = 40
    xs = np.array([WeirdNum(x) for x in np.arange(xmin, xmax, 0.1)])
    ys = np.array([WeirdNum(y) for y in np.arange(xmin, xmax, 0.1)])
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(f)(X, Y)
    Z = np.vectorize(lambda wn: float(wn.n))(Z)
    plt.contourf(Z)
    plt.contour(Z, colors="k")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # report_operations_on_random_pair()
    a = WeirdNum(random.uniform(-10, 10))
    b = WeirdNum(random.uniform(-10, 10))
    c = WeirdNum(random.uniform(-10, 10))
    d = WeirdNum(random.uniform(-10, 10))
    e = WeirdNum(random.uniform(-10, 10))
    f = WeirdNum(random.uniform(-10, 10))
    func = lambda x, y: a*(x*x) + b*(x*y) + c*(y*y) + d*x + e*y + f
    plot_function(func)
