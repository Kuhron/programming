# based on game with Nicola at King Spa on Go board

# start with some pattern (allow user to define it)
# pick a pivot point
# rotate the pattern 180 deg about that point
# overlay on existing board
# if conflict (a point is going to be both black and white), annihilate
# then ask user to make any more changes
# then ask user for new pivot
# repeat, plot each time

# dict of ordered pairs to colors (0 = nothing, 1 = black, -1 = white)


import matplotlib.pyplot as plt

class COLOR:
    N = 0  # none
    B = 1  # black
    W = -1 # white
    # note that this can also stand for Nicola Brown & Wesley

    @staticmethod
    def to_color_str(value):
        return {
            COLOR.N: None,
            COLOR.B: "k",
            COLOR.W: "w",
        }[value]


def get_user_changes():
    change_dict = {}
    print("Change format: x y v\nx = x-coord, y = y-coord, v = value\nvalues: n/0 = no color, b/+ = black, w/- = white\npress enter to stop inputting changes\n")
    while True:
        inp = input("Next change: ")
        if inp.strip() == "":
            break
        try:
            x, y, v = inp.strip().split()
        except ValueError:
            print("invalid input")
            continue
        try:
            x = int(x)
            y = int(y)
        except ValueError:
            print("invalid coords")
            continue
        if v.lower() not in "nbw0+-":
            print("invalid value")
            continue
        new_v = {"n": COLOR.N, "0": COLOR.N, "b": COLOR.B, "+": COLOR.B, "w": COLOR.W, "-": COLOR.W}[v]
        change_dict[(x, y)] = new_v
    return change_dict

def get_volley_of_user_changes(array):
    while True:
        plot_array(array)
        inp = input("Would you like to make changes? (type anything for yes, nothing for no)\n")
        if inp.strip() == "":
            break
        change_dict = get_user_changes()
        array = apply_changes(array, change_dict)

    inp = input("Would you like to pivot the array? (type anything for yes, nothing for no)\n")
    if inp.strip() == "":
        pass
    else:
        array = pivot_array(array, get_pivot())
    
    return array

def apply_changes(array, change_dict):
    points_to_remove = [k for k, v in change_dict.items() if v == COLOR.N]
    for p in points_to_remove:
        del array[p]
        del change_dict[p]
    for k, v in change_dict.items():
        x, y = k
        array[(x, y)] = v
    return array

def get_pivot():
    while True:
        inp = input("new pivot coordinates, format: x y\n")
        try:
            x, y = inp.strip().split()
            x = int(x)
            y = int(y)
            return (x, y)
        except ValueError:
            print("invalid input")
            continue

def pivot_array(array, pivot):
    def pivot_point_about_pivot(point, pivot):
        x, y = point
        xp, yp = pivot
        # 180 deg turn, just go the displacement vectors in the opposite directions
        new_x = xp - (x - xp)
        new_y = yp - (y - yp)
        return (new_x, new_y)

    def twin(p):
        return pivot_point_about_pivot(p, pivot)

    # annihilate points before doing anything else
    for point in array:
        if twin(point) in array and array[point] != array[twin(point)]:
            array[point] = COLOR.N
            array[twin(point)] = COLOR.N
    array = {k: v for k, v in array.items() if v != COLOR.N}  # avoid changing size during iteration

    new_changes = {}
    for point in array:
        assert twin(point) not in array or array[twin(point)] == array[point] # previous part should have worked
        new_changes[twin(point)] = array[point]
    array.update(new_changes)

    return array

def plot_array(array):
    xs = []
    ys = []
    cs = []
    for k, v in array.items():
        x, y = k
        xs.append(x)
        ys.append(y)
        cs.append(COLOR.to_color_str(v))
    plt.scatter(xs, ys, c=cs)
    plt.gca().set_facecolor("#e1c699")
    plt.show()


if __name__ == "__main__":
    array = {
        (0, 0): COLOR.B,
        (0, 1): COLOR.W,
        (0, 2): COLOR.B,
        (1, 1): COLOR.B,
        (1, 2): COLOR.W,
        (2, 2): COLOR.B,
    } # set initial conditions if you are playing around and don't want to type them all in again every time

    while True:
        array = get_volley_of_user_changes(array)
        plot_array(array)

