import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import Music.MusicalStructureUtil as structure


PITCH_NUMBER_TO_COLOR = [
    (255, 255,   0),  # C
    (255, 180,   0),  # K
    (  0,   0, 255),  # D
    (  0, 255, 255),  # H
    (255,   0,   0),  # E
    (  0, 120,   0),  # F
    ( 40, 255,   0),  # X
    ( 60,   0, 200),  # G
    (210,   0, 255),  # J
    (  0,   0,   0),  # A
    ( 70,  40,   0),  # R
    (120, 120, 120),  # B
]
BLACK = (0, 0, 0)


def create_rgb_array_from_pitch_classes(pitch_classes):
    pitch_numbers = [structure.pitch_class_to_number(x) for x in pitch_classes]
    colors = [PITCH_NUMBER_TO_COLOR[x] for x in pitch_numbers]
    # TODO: make black border lines between colors
    # just to get it to work: use one color
    assert len(pitch_classes) == 1
    color = colors[0]
    height = 2 * len(colors) + 1
    is_border_line = lambda y: y % 2 == 0
    array = np.zeros((height, height, 3), "uint8")
    array[:, [0, -1], :] = BLACK  # side columns
    array[-1, 1: -1, :] = BLACK  # bottom
    for i, color in enumerate(colors):
        array[2 * i, 1: -1, :] = color
        array[2 * i + 1, 1: -1, :] = BLACK
    return array


def show_image_for_pitch_classes(pitch_classes):
    array = create_rgb_array_from_pitch_classes(pitch_classes)
    im = Image.fromarray(array)
    plt.imshow(im, interpolation="none")
    plt.draw()


if __name__ == "__main__":
    pitch_classes = ["X"]
    show_image_for_pitch_classes(pitch_classes)