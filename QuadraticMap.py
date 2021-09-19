import random
import numpy as np
import matplotlib.pyplot as plt


def quadratic_map_2d(x, y, coefficients):
    a = coefficients
    return (
        a[0] + a[1] * x + a[2] * x**2 + a[3] * x * y + a[4] * y + a[5] * y**2,
        a[6] + a[7] * x + a[8] * x**2 + a[9] * x * y + a[10] * y + a[11] * y**2,
    )


def get_late_iterations(f, x0, y0, coefficients, n0, n1):
    x, y = x0, y0
    old_x, old_y = x0, y0
    for i in range(n0):
        # print(i, x, y)
        x, y = f(x, y, coefficients)
        if x == old_x and y == old_y:
            print("steady state")
            break
        old_x, old_y = x, y
    xs = [x]
    ys = [y]
    for i in range(n1 - n0):
        # print(i, x, y)
        x, y = f(x, y, coefficients)
        xs.append(x)
        ys.append(y)
        if xs[-1] == xs[-2] and ys[-1] == ys[-2]:
            print("steady state")
            break
    return xs, ys


if __name__ == "__main__":
    # want to reproduce plots like those at http://mathworld.wolfram.com/StrangeAttractor.html
    wolfram_to_number = lambda x: -1.2 + (2.4/24) * "ABCDEFGHIJKLMNOPQRSTUVWXY".index(x)
    wolfram_str_to_coefficients = lambda s: [wolfram_to_number(x) for x in s]

    # coefficients = np.random.uniform(-0.5, 0.5, size=(12,))
    wolfram_strs = [
        "AMTMNQQXUYGA", "CVQKGHQTPHTE", "FIRCDERRPVLD", "GIIETPIQRRUL", "GLXOESFTTPSV", "GXQSNSKEECTX",
        "HGUHDPHNSGOH", "ILIBVPKJWGRR", "LUFBBFISGJYS", "MCRBIPOPHTBN", "MDVAIDOYHYEA", "ODGQCNXODNYA",
        "QFFVSLMJJGCR", "UWACXDQIGKHF", "VBWNBDELYHUL", "WNCSLFLGIHGL",
    ]
    # maybe can use some heuristic measure of chaos to try to find more of these
    coefficients = wolfram_str_to_coefficients(random.choice(wolfram_strs))
    print(coefficients)

    xs, ys = get_late_iterations(quadratic_map_2d, 0, 0, coefficients, 10000, 100000)
    plt.scatter(xs, ys, c="k", marker='o', s=(72./plt.gcf().dpi)**2, alpha=0.1)
    plt.savefig("QMap.png")
