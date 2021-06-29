# show the condition values for a certain variable on a certain map
# just meant to show how these are easily accessed from the memo files


import numpy as np
import matplotlib.pyplot as plt
import os

from ImageMetadata import get_image_metadata_dict


def get_condition_array_categorical(image_name, map_variable):
    metadata = get_image_metadata_dict()[image_name]
    condition_array_dir = metadata["condition_array_dir"]

    filename = f"{image_name}_{map_variable}_condition_shorthand.txt"
    fp = os.path.join(condition_array_dir, filename)

    with open(fp) as f:
        lines = f.readlines()
    strs = [l.strip().split(",") for l in lines]
    # assume all shorthands are non-negative ints, give -1 to the absent condition
    ints = [[int(x) if x != "" else -1 for x in s] for s in strs]
    return ints


def plot_condition_array(image_name, map_variable):
    arr = get_condition_array_categorical(image_name, map_variable)
    plt.imshow(arr)
    plt.title(f"{map_variable} conditions in {image_name}")
    plt.show()


if __name__ == "__main__":
    image_names = sorted(get_image_metadata_dict().keys())
    for image_name in image_names:
        map_variable = "elevation"
        plot_condition_array(image_name, map_variable)


