# show why 12 is a good number of steps to divide the octave into
# because one of its fractions, 7/12, as a power of 2,
# does a good job approximating the pitch ratio 3/2

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    r = 3/2
    l2r = np.log2(r)
    ns = list(range(2, 25))

    # plot lines at the equal-tempered values for these octave divisions
    plt.subplot(3,1,1)
    ax1 = plt.gca()
    ax1.plot([ns[0]-1, ns[-1]+1], [l2r, l2r], c="r")  # reference pitch
    for n in ns:
        x0 = n - 1/3
        x1 = n + 1/3
        for i in range(0, n+1):
            y = i/n
            ax1.plot([x0, x1], [y, y], c="k")

    ax1.set_xticks(ns)
    ax1.set_xticklabels(ns)
    ax1.set_ylabel("fraction of octave")
    yticks = list(np.arange(-0.1, 1.2, 0.1))
    ylabels1 = yticks
    ylabels2 = [2**y for y in yticks]
    ylim = [yticks[0], yticks[-1]]
    ax1.set_ylim(ylim)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([""] + [f"{y:.1f}" for y in ylabels1[1:-1]] + [""])
    ax2 = ax1.twinx()
    ax2.set_ylabel("pitch ratio")
    ax2.set_ylim(ylim)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([""] + [f"{y:.4f}" for y in ylabels2[1:-1]] + [""])

    # now plot the error from the nearest semitone in each scale
    plt.subplot(3,1,2)
    min_frac_diffs = []
    for n in ns:
        min_frac_diff = float("inf")
        for i in range(0, n+1):
            frac = i/n
            ratio = 2**frac
            frac_diff = abs(frac - l2r)
            min_frac_diff = min(min_frac_diff, frac_diff)
        min_frac_diffs.append(min_frac_diff)
    plt.bar(ns, min_frac_diffs)
    ax = plt.gca()
    ax.set_ylabel("error as octave fraction")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)

    plt.subplot(3,1,3)
    # what proportion of the scale interval is the error?
    min_frac_diffs_scaled = [n*diff for n, diff in zip(ns, min_frac_diffs)]
    plt.bar(ns, min_frac_diffs_scaled)
    ax = plt.gca()
    ax.set_ylabel("error as scale step fraction")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)

    ax.set_xlabel("number of octave divisions")

    plt.show()
