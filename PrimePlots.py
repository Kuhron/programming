import sympy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

prime = sympy.prime  # gets nth prime, counting from prime(1) = 2
is_prime = sympy.isprime


def four_index(n):
    # special cases
    if n == 2:
        return (0, 0.5)
    elif n == 5:
        return (0, 1.5)

    # beyond this, other primes should end in 1, 3, 7, or 9 (base 10)
    endings = [1, 3, 7, 9]
    div, mod = divmod(n, 10)
    if mod not in endings:
        print("invalid four index for {}".format(n))
        return (div, -1)
    b = endings.index(mod)
    return (div, b)


def plot_four_indexes(n_rows, columns_per_row):
    column_height = 4
    y_offset_per_row = column_height + 1

    # 1 column in 1 row corresponds to 10 integers (4 of which are checked for primality)
    max_ten_to_check = n_rows * columns_per_row
    # xs_to_check = []
    # for ten in range(max_ten_to_check):
    #     xs_to_check += [10 * ten + ending for ending in [1, 3, 7, 9]]
    # xs = list(range(1, n_primes + 1))
    # assert len(xs) == n_primes
    # print("getting primes")
    # ps = [prime(x) for x in xs]

    # fis = set(four_index(p) for p in ps)
    # mx = max(a for a, b in fis)
    all_pts = [(a, b) for a in range(0, max_ten_to_check) for b in range(4)]
    len_all_pts = len(all_pts)
    print("plotting")
    for i, pt in enumerate(all_pts):
        print("{}/{} done".format(i, len_all_pts), end="\r")
        a, b = pt
        row_number, col_number = divmod(a, columns_per_row)
        # note that height of a row is 4, so we subtract 4+1 from the y value for each subsequent row, to keep the visualization clearer
        color = "black" if is_prime(10 * a + [1, 3, 7, 9][b]) else "red"
        rectangle_diameter = 1
        k = rectangle_diameter / 2
        a2 = col_number
        b2 = b - row_number * y_offset_per_row
        plt.scatter(a2, b2, color=color)  # leave this here so the points will still be plotted and the whole graph will show instead of just the [0, 1] box
        plt.gca().add_patch(Rectangle((a2 - k, b2 - k), width=rectangle_diameter, height=rectangle_diameter, color=color))
    plt.show()


if __name__ == "__main__":
    plot_four_indexes(n_rows=20, columns_per_row=40)
