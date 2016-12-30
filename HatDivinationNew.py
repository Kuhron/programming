import random
import string

import numpy as np
import matplotlib.pyplot as plt


LETTERS = string.ascii_uppercase
NUMBERS = ["1", "10", "11", "100", "101", "110", "111", "1001", "1010", "1011", "1100", "1101", "1110", "1111", "10010", "10011", "10100", "10101", "10110", "10111", "11001", "11010", "11011", "11100", "11101", "11110"]

LETTER_TO_NUMBER = dict(zip(LETTERS, NUMBERS))
NUMBER_TO_LETTER = dict(zip(NUMBERS, LETTERS))

MAPPINGS = {
    "original": dict(zip(LETTERS, LETTERS)),
    "English": dict(zip("DHOPANBEIJQRSTCFGKLMUVWXYZ", "ETAOINSHRDLUCMFWYPVBGKQJXZ")),
}


def preprocess_string(s):
    assert all([x in "01"] for x in s)

    while len(s) > 0 and s[0] == "0":
        s = s[1:]

    r = s[::-1]
    while "000" in r:
        r = r.replace("000", ",")
    s = r[::-1]

    s += "."

    while ",," in s:
        s = s.replace(",,", ",")

    while ",." in s:
        s = s.replace(",.", ".")

    return s


def split_segment(s):
    assert all([x in "01"] for x in s)

    if len(s) == 0:
        return []

    if s[0] == "0":
        raise ValueError("segment starts with 0")

    if len(s) >= 5 and s[:5] == "11111":
        return ["1111"] + split_segment(s[4:])

    if 1 <= len(s) <= 5:
        return [s]

    i = 5
    while s[i] == "0":
        i -= 1
    return [s[:i]] + split_segment(s[i:])

    raise RuntimeError("unhandled case")


def convert_string(s, mapping="original"):
    s = preprocess_string(s)
    segments = s.split(".")[0].split(",")
    segments = [split_segment(segment) for segment in segments]
    numbers = [num for split_seg in segments for num in split_seg]
    return "".join([MAPPINGS[mapping][NUMBER_TO_LETTER[x]] for x in numbers])


def plot_frequencies(message):
    counts = [message.count(x) for x in LETTERS]
    proportions = {x: counts[i] / len(message) for i, x in enumerate(LETTERS)}
    for x, y in sorted(proportions.items()):
        print("{0} : {1:.4f}".format(x, y))

    plt.bar(range(26), counts, width=1.0)
    plt.gca().set_xticks([x + 0.5 for x in range(26)])
    plt.gca().set_xticklabels(LETTERS)
    plt.show()


if __name__ == "__main__":
    original_data = "100010111010101000000010111011010010000000001011000001100000010000010010010111011000110010111000010101011000000010101000001010100000000000000011111001110110"
    assert convert_string(original_data) == "AJIIJSHEDCDDOYAKJBISIDRNPS"

    original_p = original_data.count("1") / len(original_data)

    n = 1000
    p = original_p

    s = "".join(np.random.choice(["0", "1"], n, p=[1 - p, p]))
    message = convert_string(s, mapping="original")
    # print(message)

    plot_frequencies(message)

    # when 0 and 1 are equally probable, find that there seem to be four groups, each with very similar (likely the same) probability:
    # 6.05% : DHOP
    # 5.38% : AN
    # 4.10% : BEIJQRST
    # 2.70% : CFGKLMUVWXYZ
    # can rearrange given English letter frequencies by just mapping between this order and a rough grouped order of actual frequencies