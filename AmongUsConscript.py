# conscript idea from the wire task on Among Us

import matplotlib.pyplot as plt
import random


def get_permutations(lst):
    if len(lst) <= 1:
        return [lst]
        # empty lst is the only permutation of zero-length list, 0! = 1
    else:
        res = []
        for i in range(len(lst)):
            item = lst[i]
            remainder = lst[:i] + lst[i+1:]
            perms = get_permutations(remainder)
            these = [[item] + perm for perm in perms]
            res += these
        return res


def plot_permutation(perm):
    assert len(perm) == 4
    assert sorted(perm) == [0,1,2,3]
    ys = [0, 1, 2, 3]
    colors = ["b", "r", "magenta", "y"]
    left_x = 0
    right_x = 3
    for i in range(4):
        left_y = ys[i]
        left_point = (left_x, left_y)
        right_y_index = perm[i]
        right_y = ys[right_y_index]
        right_point = (right_x, right_y)
        print(left_point, right_point)
        xs_plot = [left_x, right_x]
        ys_plot = [left_y, right_y]
        plt.plot(xs_plot, ys_plot, linewidth=8, color=colors[i])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', 'box')  # https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/axis_equal_demo.html
    plt.gca().set_facecolor("black")
    plt.axis("off")
    # plt.show()


def plot_all_permutations():
    perms = get_permutations([0, 1, 2, 3])
    n_rows = 4
    n_cols = 6
    for i in range(24):
        # row_num, col_num = divmod(i, n_cols)
        plt.subplot(n_rows, n_cols, i+1)
        plot_permutation(perms[i])
        plt.title(i)



if __name__ == "__main__":
    perms = get_permutations([0,1,2,3])
    assert len(perms) == 24
    perm = random.choice(perms)
    plot_permutation(perm)
    plt.show()

    plot_all_permutations()
    plt.show()
