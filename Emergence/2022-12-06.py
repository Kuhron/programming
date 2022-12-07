# idea from playing with an Excel sheet

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata


def fill_arr(a, b, n, f):
    # a is the column headers, the first seed value
    # b is the row headers, the second seed value
    arr = np.zeros((n+1, n+1))
    arr[0,0] = np.nan
    arr[0,1:] = a
    arr[1:,0] = b
    for i in range(1, n+1):
        for j in range(1, n+1):
            U = arr[i-1, j]
            L = arr[i, j-1]
            val = f(L, U)
            arr[i,j] = val
    return arr


if __name__ == "__main__":
    n = 400
    fs = {
        # "f1": lambda L, U: (L*U + L/U) % (L+U),  # original one is still the most interesting one
        # "f2": lambda L, U: (L*U + L/U) % (U/L),
        # "f3": lambda L, U: (L*U) % (L+U),
        # "f4": lambda L, U: (L**2 + U**2) % (L*U + L + U)  # trying to find something else interesting
    }

    for fname, f in fs.items():
        for a in range(1, 13):
            for b in range(1, 13):
                print(fname, a, b)
                arr = fill_arr(a, b, n, f)
                ranks = rankdata(arr, method="min").reshape((n+1,n+1))
                plt.imshow(ranks, cmap="CMRmap")
                plt.colorbar()
                plt.savefig(f"Images/2022-12-06/{fname}_{a}_{b}.png")
                plt.gcf().clear()
