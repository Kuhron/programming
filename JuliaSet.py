import numpy as np
import matplotlib.pyplot as plt
import random


def plot_julia_set(c, resolution, iterations):
    f = lambda z: z**2 + c
    xs = np.linspace(-2, 2, resolution)
    ys = np.linspace(-2, 2, resolution)
    # zs = []
    divergence_times = [[np.nan for x in xs] for y in ys]
    # array should go (from upper left): rows are decreasing y, columns are increasing x
    keep_condition = lambda z: abs(z) <= 2
    for y_i, y in enumerate(ys[::-1]):
        print(f"y = {y:.4f} (step {y_i}/{resolution//2})")
        # zs_row = []
        ts_row = []
        for x in xs:
            z = x + 1j*y
            diverged = False
            for i in range(iterations):
                z = f(z)
                if not keep_condition(z):
                    # zs_row.append(np.nan)
                    ts_row.append(i)
                    diverged = True
                    break
            if not diverged:
                # zs_row.append(z)
                ts_row.append(np.inf)
        # zs.append(zs_row)
        divergence_times[y_i] = ts_row
        divergence_times[resolution - 1 - y_i] = ts_row[::-1]  # 180 degree symmetry
        if y <= 0:
            break  # take advantage of 180 degree rotational symmetry and only do the top half

    # zs = np.array(zs)
    # if (np.isnan(zs)).all():  # this IS true for complex nan, i.e. np.nan+0j
    #     print("all points diverged")
    #     return
    ts = np.array(divergence_times)
    assert not np.isnan(ts).any()
    plt.imshow(ts)
    print(f"c = {c}")
    plt.title(f"c = {c}")
    plt.show()


def get_c_in_disc():
    while True:
        c = random.uniform(-2, 2) + 1j*random.uniform(-2, 2)
        if abs(c) <= 2:
            return c


if __name__ == "__main__":
    resolution = 10001
    iterations = 10000
    # c = get_c_in_disc()
    c = -0.7807036373151086+0.2674077984300518j
    print(f"c = {c}")
    plot_julia_set(c, resolution, iterations)
