import random
import string
import numpy as np

from InfiniteMonkeys import WordTree


LETTERS = string.ascii_uppercase
FREQUENCIES = [
    9, 2, 2, 4, 12,
    2, 3, 2, 9, 1,
    1, 4, 2, 6, 8,
    2, 1, 6, 4, 6,
    4, 2, 2, 1, 2, 1
]
VALUES = [
    1, 3, 3, 2, 1, 
    4, 2, 4, 1, 8, 
    5, 1, 3, 1, 1, 
    3, 10, 1, 1, 1,
    1, 4, 4, 8, 4, 10
]
LTV = dict(zip(LETTERS, VALUES))
NORM_FREQS = [x / sum(FREQUENCIES) for x in FREQUENCIES]


def get_dictionary():
    with open("ScrabbleDictionary.txt") as f:
        lines = f.readlines()
    tree = WordTree((x.strip() for x in lines))
    return tree

def get_value(letter):
    return LTV[letter]

def get_letters():
    return [np.random.choice(list(LETTERS), p=NORM_FREQS) for i in range(7)]

def print_letters(letters):
    s1 = s2 = ""
    for letter in letters:
        s1 += str(letter).rjust(3)
        s2 += str(get_value(letter)).rjust(3)
    print()
    print(s1)
    print(s2)

def word_is_in_tray(word, letters):
    for c in set(word):
        if word.count(c) > letters.count(c):
            return False
    return True

def score(word):
    return sum(get_value(c) for c in word)


if __name__ == "__main__":
    dictionary = get_dictionary()
    while True:
        letters = get_letters()
        while True:
            print_letters(letters)
            word = input("word (press enter to get new letters): ").upper()
            if len(word) == 0:
                break
            if not all(x in LETTERS for x in word):
                print("invalid word; use letters only\n")
                continue
            if word in dictionary:
                print("That is a real word! ", end="")
                if word_is_in_tray(word, letters):
                    print("Good job! It is worth {} points.".format(score(word)))
                else:
                    print("Unfortunately you cannot make it with your letters.")
            else:
                print("That is not a real word. ")
            print()
