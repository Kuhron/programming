# encode a set card by a CVCV word
# w~v~beta

import random


class Feature:
    @classmethod
    def random(cls):
        return random.choice(cls.ARRAY)


class Number(Feature):
    ONE = 0
    TWO = 1
    THREE = 2
    ARRAY = [ONE, TWO, THREE]
    WORDS = ["one", "two", "three"]
    SOUNDS = ["w", "s", "k"]


class Shading(Feature):
    EMPTY = 0
    SHADED = 1
    SOLID = 2
    ARRAY = [EMPTY, SHADED, SOLID]
    WORDS = ["empty", "shaded", "solid"]
    SOUNDS = ["i", "a", "u"]


class Color(Feature):
    RED = 0
    BLUE = 1  # blue in Triple Play or purple in Set
    GREEN = 2
    ARRAY = [RED, BLUE, GREEN]
    WORDS = ["red", "blue", "green"]
    SOUNDS = ["m", "r", "ng"]  # engma-g cluster, not just engma


class Shape(Feature):
    SQUIGGLE = 0  # line in Triple Play or squiggle in Set
    DIAMOND = 1
    CIRCLE = 2  # circle in Triple Play or oval in Set
    ARRAY = [SQUIGGLE, DIAMOND, CIRCLE]
    WORDS = ["squiggle", "diamond", "circle"]
    SOUNDS = ["i", "a", "u"]


class Card:
    cls_array = [Number, Shading, Color, Shape]

    def __init__(self, number, shading, color, shape):
        self.number = number
        self.shading = shading
        self.color = color
        self.shape = shape
        self.attr_array = [number, shading, color, shape]

    def get_words_and_sounds(self):
        words_arr = []
        sounds = ""
        for cls, attr in zip(Card.cls_array, self.attr_array):
            word = cls.WORDS[attr]
            sound = cls.SOUNDS[attr]
            words_arr.append(word)
            sounds += sound
        return " ".join(words_arr), sounds

    @staticmethod
    def random():
        attrs = []
        for cls in Card.cls_array:
            attr = cls.random()
            attrs.append(attr)
        return Card(*attrs)


if __name__ == "__main__":
    card = Card.random()
    words, sounds = card.get_words_and_sounds()
    print(words)
    print(sounds)
