# another idea for Language and Cognition experiment, give people colors that are fully saturated and full-value, with the only variable being hue (which is topologically a circle)
# they have to do a communication game to name the stimuli, BUT unbeknownst to the participants, the hue is rotated by some constant random amount between them
# so e.g. if it's + 1/3, then the describer might say "red" but the listener sees yellow; this could be a way to prevent them from cheating and using real language terms
# and it would be able to control for the pre-existing linguistic color categories that the speakers have, e.g. they make finer distinctions in the red-orange-yellow area than they do in the green-blue area, but this is shifted among them so those biases won't "line up" and new category boundaries may result
# the options given as potential answers can also be easily measured in terms of how far off the answer was, so we can quantify the accuracy better than just right/wrong


import random
import os
import numpy as np
import matplotlib.pyplot as plt
# import colorsys
from matplotlib.colors import hsv_to_rgb


def show_random_color():
    hue = random.random()
    show_color(hue)


def show_color(hue):
    rgb = hsv_to_rgb([hue, 1, 1])
    arr = [[rgb]]
    plt.imshow(arr)
    plt.show()


def show_color_wheels(offset_of_second_wheel):
    xs = np.arange(0, 0.2, 0.01)
    ys = np.arange(0, 1, 0.01)  # using arange instead of linspace so the width/height will be on the same scale
    X, Y = np.meshgrid(xs, ys)
    get_z0 = lambda y: y
    get_z1 = lambda y: (y + offset_of_second_wheel) % 1
    get_ones = lambda y: 1
    ONE = np.vectorize(get_ones)(Y)
    Z0 = np.vectorize(get_z0)(Y)
    Z1 = np.vectorize(get_z1)(Y)

    HSV0 = np.stack([Z0, ONE, ONE], axis=-1)
    HSV1 = np.stack([Z1, ONE, ONE], axis=-1)
    RGB0 = hsv_to_rgb(HSV0)
    RGB1 = hsv_to_rgb(HSV1)

    human_color_names = get_human_names_for_colors()
    boundaries_by_color_name = get_boundaries_by_color_name()
    label_positions_by_color_name = get_label_positions_by_color_name()
 
    plt.subplot(1, 2, 1)
    plt.imshow(RGB0)
    for color, boundary in boundaries_by_color_name.items():
        y = boundary - 0.5
        plt.plot([0,19.5],[y,y],c="k")
    for color, label_position in label_positions_by_color_name.items():
        y = label_position
        plt.text(22,y,color)

    plt.subplot(1, 2, 2)
    plt.imshow(RGB1)
    for color, boundary in boundaries_by_color_name.items():
        y = (boundary - offset_of_second_wheel*100 - 0.5) % 100  # need to SUBTRACT the offset to get the apparent y
        plt.plot([0,19.5],[y,y],c="k")
    for color, label_position in label_positions_by_color_name.items():
        y = (label_position - offset_of_second_wheel*100) % 100
        plt.text(22,y,color)


    plt.title(f"offset {offset_of_second_wheel:.2f}")

    plt.show()


def get_user_input_names_for_colors():
    hues = np.arange(0, 1, 0.01)
    fp = "HueRotationColorNames.txt"
    if not os.path.exists(fp):
        with open(fp, "w") as f:
            for i in range(100):
                f.write(f"{i} = \n")
    order = list(range(100))
    random.shuffle(order)
    for i in order:
        with open(fp) as f:
            lines = f.readlines()
        line = lines[i]
        assert line.startswith(f"{i} = ")

        i_str, past_name = line.split(" = ")
        if past_name.strip() == "":
            hue = i * 0.01
            show_color(hue)
            name = input("please name the color you just saw: ").strip()
            new_line = f"{i} = {name}\n"
            lines = lines[:i] + [new_line] + lines[i+1:]
            with open(fp, "w") as f:
                for line in lines:
                    f.write(line)


def get_human_names_for_colors():
    with open("HueRotationColorNames.txt") as f:
        lines = f.readlines()
    d = {}
    for line in lines:
        i_str, name = line.strip().split(" = ")
        i = int(i_str)
        d[i] = name
    return d


def get_boundaries_by_color_name():
    # just did this manually based on my naming answers
    # lower boundary, i.e. the first hue that gets this name
    return {
        "red":96,
        "orange":4,
        "yellow":12,
        "green":23,
        "sky blue":44,
        "blue":56,
        "purple":74,
        "pink":81,
    }


def get_label_positions_by_color_name():
    # also did this manually
    return {
        "red": 0,
        "orange": 8,
        "yellow": 18,
        "green": 33,
        "sky blue": 50,
        "blue": 65,
        "purple": 78,
        "pink": 88,
    }


if __name__ == "__main__":
    human_color_names = get_human_names_for_colors()
    boundaries_by_color_name = get_boundaries_by_color_name()
    label_positions_by_color_name = get_label_positions_by_color_name()
    # show_random_color()
    show_color_wheels(offset_of_second_wheel=0.01*random.randint(0,99))
