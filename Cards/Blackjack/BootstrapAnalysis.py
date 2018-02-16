import os
import numpy as np  # really do we have to import everything in numpy just to get a range with step < 1?
import matplotlib.pyplot as plt


def get_shoes_with_tc(tc):
    tc = round(tc, 1)
    shoe_fps = os.listdir("Bootstraps/")
    total = len(shoe_fps)
    search_str = "_{tc}_".format(tc=tc)
    shoe_fps = [x for x in shoe_fps if search_str in x]
    n = len(shoe_fps)
    return n, total, n/total


if __name__ == "__main__":
    tcs = []
    ns = []

    for tc in np.arange(-15, 15, 0.1):
        if abs(tc) < 1e-3:  # i hate floats
            tc = 0
        n, total, frac = get_shoes_with_tc(tc)
        print(tc, n, frac)
        tcs.append(tc)
        ns.append(n)
    plt.plot(tcs, ns)
    plt.show()
