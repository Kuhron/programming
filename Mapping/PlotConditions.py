# show the condition values for a certain variable on a certain map
# just meant to show how these are easily accessed from the memo files


import numpy as np
import matplotlib.pyplot as plt
import os

from ImageMetadata import get_image_metadata_dict
from LoadMapData import get_condition_array_categorical


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


