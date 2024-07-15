import sympy
import numpy as np
import matplotlib.pyplot as plt


def plot_path_of_numbers(n_min, n_max, angle_deg, show=True):
    xs, ys = get_path_of_numbers(n_min, n_max, angle_deg)
    plt.plot(xs, ys)
    plt.gca().set_aspect("equal")

    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    sq_inches = 16*24
    n_boxes = x_range * y_range  # units of area to determine how to scale figure size
    sq_inches_per_box = sq_inches / n_boxes
    inches_per_unit = sq_inches_per_box ** 0.5

    plt.gcf().set_size_inches((x_range*inches_per_unit, y_range*inches_per_unit))
    plt.savefig(f"PrimeTurtleImages/{n_min}_{n_max}_{angle_deg}.png")
    if show:
        plt.show()
    plt.clf()


def get_path_of_numbers(n_min, n_max, angle_deg):
    xs = []
    ys = []
    xy = np.array([0, 0])
    dxy = np.array([1, 0])

    if angle_deg == 90:
        rotation_matrix = np.array([[0, -1], [1, 0]])  # stupid float crap
    else:
        theta = angle_deg * np.pi/180
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    for n in range(n_min, n_max):
        if n % 100000 == 0:
            print(f"{n = }/{n_max}")
        if sympy.isprime(n):
            # rotate the direction vector
            dxy = rotation_matrix @ dxy
        steps = 1
        xy = xy + steps*dxy  # if you do += then you get stupid numpy float casting error
        x,y = xy
        xs.append(x)
        ys.append(y)
    return xs, ys


if __name__ == "__main__":
    n_min = 1
    n_max = 10**5
    angle_deg = 80

    for angle_deg in range(5, 180, 5):
        print(angle_deg)
        plot_path_of_numbers(n_min, n_max, angle_deg, show=False)



# questions:
# - varying the turn angle, given a number n, what point does n land at? what angle is n at in polar coordinates? do these vary continuously (I'd expect they do, but the larger n is the larger the impact of changing the angle)
# - 
