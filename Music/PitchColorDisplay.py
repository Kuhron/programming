import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import Music.MusicalStructureUtil as structure
import Music.MusicParser as parser
import Music.WavUtil as wav


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
    (255, 255, 255),  # A
    ( 70,  40,   0),  # R
    (120, 120, 120),  # B
]
BLACK = (0, 0, 0)


def create_rgb_array_from_pitch_classes(pitch_classes):
    pitch_numbers = [structure.pitch_class_to_number(x) for x in pitch_classes]
    colors = [PITCH_NUMBER_TO_COLOR[x] for x in pitch_numbers]
    height = 2 * len(colors) + 1
    is_border_line = lambda y: y % 2 == 0
    array = np.zeros((height, height, 3), "uint8")
    array[:, [0, -1], :] = BLACK  # side columns
    array[-1, 1: -1, :] = BLACK  # bottom
    for i, color in enumerate(colors):
        array[2 * i, 1: -1, :] = BLACK
        array[2 * i + 1, 1: -1, :] = color
    return array


def show_image_for_note_or_chord(x):
    if type(x) is structure.Note:
        pitch_classes = [x.pitch_class]
    elif type(x) is structure.Chord:
        pitch_classes = [y.pitch_class for y in sorted(x.notes, key=lambda x: x.midi_pitch_number, reverse=True)]
    elif type(x) is structure.Rest:
        pitch_classes = []
    else:
        raise TypeError("argument must be Note, Chord, or Rest; got {}".format(type(x)))

    array = create_rgb_array_from_pitch_classes(pitch_classes)
    im = Image.fromarray(array)
    plt.imshow(im, interpolation="none")
    plt.show()
    plt.pause(x.duration.duration_seconds)


def show_images_for_notes(notes):
    for x in notes:
        show_image_for_note_or_chord(x)


if __name__ == "__main__":
    plt.ion()
    res = parser.parse_file("Music\\MusicParserTestInput.txt", parser.TEMPO)
    signal = wav.get_signal_from_notes(res)
    wav.send_signal_to_audio_out(signal)
    show_images_for_notes(res)
    plt.pause(10)