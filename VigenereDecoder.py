import random
import string
import numpy as np
import matplotlib.pyplot as plt

abc = string.ascii_lowercase
number_to_letter = {i: x for i, x in enumerate(abc)}
letter_to_number = {x: i for i, x in enumerate(abc)}


def get_number_from_letter(x):
    return letter_to_number.get(x)


def get_letter_from_number(x):
    return number_to_letter.get(x)


def get_letter_frequencies():
    fp = "/home/wesley/programming/letter_frequencies.csv"
    d = {}
    with open(fp) as f:
        lines = f.readlines()
    for line in lines:
        letter, freq = line.strip().split(",")
        assert letter not in d
        d[letter] = float(freq)
    return d


def add_letters(a, b):
    na = get_number_from_letter(a)
    nb = get_number_from_letter(b)
    n = na + nb
    n %= len(abc)
    return abc[n]


def viginere_encode(message, key):
    assert all(x in abc for x in key), "invalid key: {}".format(key)
    message = message.lower()
    key_gen = key_repeat_generator(key)
    res = ""
    for mc in message:
        kc = next(key_gen)
        if mc in abc:
            new = add_letters(mc, kc)
            res += new
        else:
            res += mc
            # use up the key letter anyway, it zips one to one regardless of if the char is in the alphabet or not
    return res


def key_repeat_generator(key):
    while True:
        for x in key:
            yield x


def get_random_key():
    return "".join(random.choice(abc) for i in range(random.randint(5,15)))


def get_random_message():
    fp = "/home/wesley/programming/moby_dick.txt"
    with open(fp) as f:
        contents = f.read()
    n = len(contents)
    message_len = random.randint(100,400)
    min_i = 0
    max_i = n - message_len - 1
    start = random.randint(min_i, max_i)
    end = start + message_len
    return contents[start:end]


if __name__ == "__main__":
    key = get_random_key()
    # message = "thisxisxaxsecretxmessagexgoodxluckxdecodingxitxxhopefullyxthexfactxthatxcertainxwordsxandxngramsxarexmorexcommonxwillxmakexitxeasierxtoxinferxsomethingxaboutxwhichxkeysxarexmorexlikelyxx"
    # message = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # test that the key repeats correctly
    message = get_random_message()
    print(viginere_encode(message, key))

