import random
import string
import math
import numpy as np
import matplotlib.pyplot as plt


class WordSearchDataStructure:
    # meant for efficiently checking whether the lines in word_frequencies_en_full are about a given set of words
    def __init__(self, words):
        allowed_chars = string.ascii_uppercase
        assert all(all(c in allowed_chars for c in w) for w in words), "disallowed chars"
        self.words = words
        self.path_dict = self.create_path_dict()

    def create_path_dict(self):
        # want to query a line by looking at first char
        # if no word in our set has that, then stop and return no match
        # if some words do, then look at second char, see if words in our set *which begin with the first letter* have that second char, etc.
        # make a tree path dictionary to accomplish this
        # end of a path is a value of empty dict
        # so word "tree" looks like: {"T": {"R": {"E": {"E": {}}}}}
        # then add word "try" to get: {"T": {"R": {"E": {"E": {}}, "Y": {}}}}
        d = {}
        for w in self.words:
            d = WordSearchDataStructure.add_word_to_path_dict(w, d)

        check1 = sorted(set(self.words))
        check2 = sorted(set(WordSearchDataStructure.get_words_from_path_dict(d)))
        # print(check1)
        # print(check2)
        assert check1 == check2
        return d

    @staticmethod
    def add_word_to_path_dict(w, d):
        w = w + "#"  # need the word boundary char to say when words end
        sub_d = d
        for c in w:
            if c not in sub_d:
                sub_d[c] = {}
            sub_d = sub_d[c]
        # hopefully this modification of the element dictionaries by reference will work
        return d

    @staticmethod
    def get_words_from_path_dict(d):
        if d == {}:
            return [""]  # base case
        words = []
        for char in d:
            sub_d = d[char]
            # print(char, sub_d)
            sub_words = WordSearchDataStructure.get_words_from_path_dict(sub_d)
            # print(f"sub_words: {sub_words}")
            # input("A")
            words_this_char = [char + w.replace("#", "") for w in sub_words]
            words += words_this_char
        return words

    def contains(self, word):
        d = self.path_dict
        for c in word:
            if c not in d:
                return False
            else:
                d = d[c]
        # now check that a path ends here and it's not just a sub-word with continuations (e.g. "continu" would be one such false positive if we don't check word ending)
        return "#" in d


def get_all_words(length):
    words = set()

    with open("cmudict.txt") as f:
        lines = f.readlines()
    words1 = [l.split()[0] for l in lines]
    allowed_chars = string.ascii_uppercase
    words1 = [w.upper() for w in words1]
    words1 = ["".join(c for c in w if c in allowed_chars) for w in words1]
    words1 = [w for w in words1 if len(w) == length]
    words |= set(words1)

    # check also other words files
    with open("enwiki-20190320-words-frequency.txt") as f:  # credit to https://github.com/IlyaSemenov/wikipedia-word-frequency/blob/master/results/enwiki-20190320-words-frequency.txt
        lines = f.readlines()
    words2 = set()
    for line in lines:
        try:
            word, freq = line.strip().split()
        except ValueError:
            continue
        word = word.upper()
        if len(word) == length and all(c in allowed_chars for c in word):
            words2.add(word)
    new_words = words2 - words
    # print(f"got {len(new_words)} new words in file 2")
    words |= words2

    assert all(len(w) == length for w in words), [w for w in words if len(w) != length]
    return words


def get_frequencies(words):
    # look at a corpus
    frequency_file = "enwiki-20190320-words-frequency.txt"
    with open(frequency_file) as f:
        lines = f.readlines()
    word_search = WordSearchDataStructure(words)
    counts = {}
    for line in lines:
        try:
            word, count = line.strip().split()
        except ValueError:
            continue
        word = word.upper()
        if word_search.contains(word):
            # assert word not in frequencies, f"conflict with word {word}, old freq {frequencies[word]}, new {int(freq)}"
            if word in counts:
                counts[word] += int(count)
            else:
                counts[word] = int(count)
    total_count = sum(counts.values())
    # normalize them
    frequencies = {k: v/total_count for k,v in counts.items()}
    return frequencies


def x01_to_nelda(x):
    if not (0 <= x <= 1):
        raise ValueError("x must be such that 0 <= x <= 1")
    log = math.log10(x) if x != 0 else -np.inf
    if log == 0:
        return log  # just so we don't get negative zero because it's annoying
    else:
        return -1 * log


def nelda_to_x01(d):
    return 10 ** (-d)


def get_word_history():
    fp = "wordle_history.txt"
    with open(fp) as f:
        lines = f.readlines()
    d = {}
    for l in lines:
        n_str, w = l.strip().split(" = ")
        n = int(n_str)
        d[n] = w
    return d


def get_allowed_words():
    fp = "wordle_list_all.txt"
    with open(fp) as f:
        lines = f.readlines()
    words = [l.strip() for l in lines]
    return words


def report_frequencies(frequencies, printing=True, plot=True):
    max_frequency = max(frequencies.values())
    assert max_frequency == 1, max_frequency
    neldas = []
    for w, freq in sorted(frequencies.items(), key=lambda kv: kv[1], reverse=True):
        norm_freq = freq / max_frequency
        nelda = x01_to_nelda(norm_freq)
        neldas.append(nelda)
        if printing:
            print(f"{w} {norm_freq} (nelda {nelda})")
            input()
    if plot:
        plt.hist(neldas, bins=100)
        plt.yscale("log")
        plt.show()


def report_frequencies_of_words(words, frequencies, x01=True, nelda=True, plot=True):
    neldas = []
    for w in words:
        freq = frequencies.get(w, 0)
        nelda = x01_to_nelda(freq)
        if np.isfinite(nelda):
            neldas.append(nelda)
        s = w
        if x01:
            s += f" {freq:.12f}"
            if nelda:  # if also
                s += ","
        if nelda:
            s += f" nelda {nelda:.4f}"
        print(s)
    if plot:
        plt.hist(neldas, bins=100)
        plt.yscale("log")
        plt.show()


def report_frequencies_of_letters_in_positions(words, frequencies, word_type_str):
    # how common is each letter in each position
    # and how common is each position for each letter
    counts = {}
    for position in range(1, 6):
        i = position - 1
        counts[position] = {"equal": {c: 0 for c in string.ascii_uppercase}, "freq": {c: 0 for c in string.ascii_uppercase}}
        for w in words:
            c = w[i]
            freq = frequencies.get(w, 0)
            counts[position]["equal"][c] += 1  # counting each word equally
            counts[position]["freq"][c] += freq  # counting based on frequency, more frequent words get bigger weight
        tups_equal = sorted(counts[position]["equal"].items(), key=lambda kv: kv[1], reverse=True)
        tups_freq = sorted(counts[position]["freq"].items(), key=lambda kv: kv[1], reverse=True)
        letters_equal = [tup[0] for tup in tups_equal]
        letters_freq = [tup[0] for tup in tups_freq]
        ns_equal = [tup[1] for tup in tups_equal]
        ns_freq = [tup[1] for tup in tups_freq]
        total_n_equal = sum(ns_equal)
        total_n_freq = sum(ns_freq)
        ns_equal = [n/total_n_equal for n in ns_equal]
        ns_freq = [n/total_n_freq for n in ns_freq]

        plot_name = f"letter frequencies in position {position} for {word_type_str}"
        plt.subplot(2,1,1)
        plt.title(plot_name)
        plt.bar(letters_equal, ns_equal, label="words equally weighted")
        plt.legend()
        plt.subplot(2,1,2)
        plt.bar(letters_freq, ns_freq, label="words weighted by frequency")
        plt.legend()
        plt.gcf().set_size_inches(6,4)
        plt.savefig(f"{plot_name}.png")
        plt.clf()

    # now also show how common each position is given a letter
    for c in string.ascii_uppercase:
        positions = range(1, 6)
        counts_equal = [counts[position]["equal"][c] for position in positions]
        counts_freq = [counts[position]["freq"][c] for position in positions]
        total_count_equal = sum(counts_equal)
        total_count_freq = sum(counts_freq)
        ns_equal = [n/total_count_equal for n in counts_equal]
        ns_freq = [n/total_count_freq for n in counts_freq]
        
        plot_name = f"position frequencies for letter {c} for {word_type_str}"
        plt.subplot(2,1,1)
        plt.title(plot_name)
        plt.bar(positions, ns_equal, label="words equally weighted")
        plt.legend()
        plt.subplot(2,1,2)
        plt.bar(positions, ns_freq, label="words weighted by frequency")
        plt.gcf().set_size_inches(6,4)
        plt.savefig(f"{plot_name}.png")
        plt.clf()


def query_words(allowed_words, frequencies):
    # user interaction
    while True:
        w = input("type a word: ").strip().upper()
        if w not in allowed_words:
            print("that word is not accepted")
        else:
            freq = frequencies.get(w, 0)
            nelda = x01_to_nelda(freq)
            print(f"that word is accepted, frequency nelda {nelda:.4f}")
        print()


if __name__ == "__main__":
    length = 5
    words = get_all_words(length)
    frequencies = get_frequencies(words)
    # report_frequencies(frequencies, printing=False, plot=True)

    allowed_words = get_allowed_words()
    assert all(len(w) == length for w in allowed_words)

    word_history = get_word_history()
    path_history = {
        201: {
            "w": ["SHIFT", "CANDY", "ROUGE", "BLIMP", "BANAL"],
            "n": ["STEAM", "STRAW", "ILIAD", "BANAL"],
        },
        202: {
            "w": ["STAIN", "CHORE", "PLUMY", "SLUMP"],
            "n": ["LINED", "BLACK", "PLOWS", "SLURP", "SLUMP"],
        },
        203: {
            "w": ["STAIR", "HOUND", "FLECK", "CRANK"],
            "n": ["TARES", "CHAIR", "CRAZY", "CRAWL", "CRACK", "CRANK"],
        },
        204: {
            "w": ["STAIN", "CHORE", "LUMPY", "GORGE"],
            "n": ["TARES", "BLACK", "ERROR", "HORDE", "FORGE", "GORGE"],
        },
    }

    # report_frequencies_of_words(word_history.values(), frequencies, x01=False, nelda=True)
    # report_frequencies_of_words(allowed_words, frequencies, x01=False, nelda=True)

    report_frequencies_of_letters_in_positions(allowed_words, frequencies, word_type_str="allowed words")
    report_frequencies_of_letters_in_positions(word_history.values(), frequencies, word_type_str="past solutions up to puzzle #{max(word_history.keys())}")

    # query_words(allowed_words, frequencies)
