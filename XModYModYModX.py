import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    if x == 0 or y == 0:
        return -1
    elif y % x == 0:
        return -1
    return (x % y) % (y % x)


def get_result(n):
    return np.array([[f(x, y) for x in range(2, n)] for y in range(2, n)])


a = get_result(1000)
plt.imshow(a, interpolation="None")
plt.colorbar()
plt.show()