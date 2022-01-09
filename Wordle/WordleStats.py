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


class Color:
    WRONG = 0
    MISPLACED = 1
    RIGHT = 2


def evaluate_word(guess, solution):
    res = [None] * 5
    for c in set(guess):
        indices_in_guess = [i for i,x in enumerate(guess) if x == c]
        indices_in_solution = [i for i,x in enumerate(solution) if x == c]
        # print("in evaluate_word", c, indices_in_guess, indices_in_solution)
        # first, if any of them are green, mark those and remove those indices
        green_indices = set(indices_in_guess) & set(indices_in_solution)
        for i in green_indices:
            res[i] = Color.RIGHT
            indices_in_guess.remove(i)
            indices_in_solution.remove(i)
        # then, as long as there are still some left in the solution, mark the first misplaced one as yellow
        assert set(indices_in_guess) & set(indices_in_solution) == set(), "missed some green index removal"
        indices_in_guess = sorted(indices_in_guess)
        yellow_indices = indices_in_guess[:len(indices_in_solution)]
        # if you run out of this char in solution, then mark the rest gray
        gray_indices = indices_in_guess[len(indices_in_solution):]
        for i in yellow_indices:
            res[i] = Color.MISPLACED
        for i in gray_indices:
            res[i] = Color.WRONG
    assert not any(x is None for x in res)
    return res


def test_evaluate_word():
    arr_error = evaluate_word("ERROR", "GORGE")
    assert arr_error == [1,0,2,1,0], arr_error
    arr_doggy = evaluate_word("DOGGY", "GORGE")
    assert arr_doggy == [0,2,1,2,0], arr_doggy
    arr_aggro = evaluate_word("AGGRO", "GORGE")
    assert arr_aggro == [0,1,1,1,1], arr_aggro
    arr_ardor = evaluate_word("ARDOR", "GORGE")
    assert arr_ardor == [0,1,0,1,0], arr_ardor  # it won't mark both Rs as yellow because there's only one R in the solution
    print("passed test_evaluate_word")


def get_all_words(length):
    words = set()

    with open("/home/wesley/programming/cmudict.txt") as f:
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


def x01_weighting_function(x, k):
    # a curve in the box [0,1]*[0,1] that is above or at the y=x line
    # imagine a pin being pulled at the midpoint of the line
    # for k=0, the pin is all the way at the upper left, so f(0) = 1 and in fact f(x) is 1 for all x
    # for k=1, the pin is in the middle, so f(0.5) = 0.5, and in fact f(x) = x for all x
    # the pin's x coordinate is linear in k, such that x_pin = k/2

    # so this is used with x = word frequency to get some different weightings by frequency
    # k=0 is equal weighting regardless of word frequency
    # k=1 is weighting by word frequency directly
    # intermediate values of k approximate logarithm-like underweighting of higher frequency words while still weighting them more than lower-frequency words (just not as much more)
    c = (k / (k-2)) ** 2
    return x / (x + c*(1-x))


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


def plot_frequencies_of_letters_in_positions(words, frequencies, word_type_str):
    # how common is each letter in each position
    # and how common is each position for each letter
    counts = {}
    for position in range(1, 6):
        i = position - 1
        counts[position] = {weight_type: {c: 0 for c in string.ascii_uppercase} for weight_type in ["equal", "freq", "k=0.2"]}
        for w in words:
            c = w[i]
            freq = frequencies.get(w, 0)
            weighted_freq = x01_weighting_function(freq, k=0.2)
            counts[position]["equal"][c] += 1  # counting each word equally
            counts[position]["freq"][c] += freq  # counting based on frequency, more frequent words get bigger weight
            counts[position]["k=0.2"][c] += weighted_freq  # counting based on frequency while underweighting higher frequencies but preserving order
        tups_equal = sorted(counts[position]["equal"].items(), key=lambda kv: kv[1], reverse=True)
        tups_freq = sorted(counts[position]["freq"].items(), key=lambda kv: kv[1], reverse=True)
        tups_k02 = sorted(counts[position]["k=0.2"].items(), key=lambda kv: kv[1], reverse=True)
        letters_equal = [tup[0] for tup in tups_equal]
        letters_freq = [tup[0] for tup in tups_freq]
        letters_k02 = [tup[0] for tup in tups_k02]
        ns_equal = [tup[1] for tup in tups_equal]
        ns_freq = [tup[1] for tup in tups_freq]
        ns_k02 = [tup[1] for tup in tups_k02]
        total_n_equal = sum(ns_equal)
        total_n_freq = sum(ns_freq)
        total_n_k02 = sum(ns_k02)
        ns_equal = [n/total_n_equal for n in ns_equal]
        ns_freq = [n/total_n_freq for n in ns_freq]
        ns_k02 = [n/total_n_k02 for n in ns_k02]

        plot_name = f"letter frequencies in position {position} for {word_type_str}"
        plt.subplot(3,1,1)
        plt.title(plot_name)
        plt.bar(letters_equal, ns_equal, label="k=0 (equal)")
        plt.legend()
        plt.subplot(3,1,2)
        plt.bar(letters_k02, ns_k02, label="k=0.2")
        plt.legend()
        plt.subplot(3,1,3)
        plt.bar(letters_freq, ns_freq, label="k=1 (frequency)")
        plt.legend()
        plt.gcf().set_size_inches(9,9)
        plt.savefig(f"Images/{plot_name}.png")
        plt.clf()

    # now also show how common each position is given a letter
    for c in string.ascii_uppercase:
        positions = range(1, 6)
        counts_equal = [counts[position]["equal"][c] for position in positions]
        counts_freq = [counts[position]["freq"][c] for position in positions]
        counts_k02 = [counts[position]["k=0.2"][c] for position in positions]
        total_count_equal = sum(counts_equal)
        total_count_freq = sum(counts_freq)
        total_count_k02 = sum(counts_k02)
        ns_equal = [n/total_count_equal for n in counts_equal]
        ns_freq = [n/total_count_freq for n in counts_freq]
        ns_k02 = [n/total_count_k02 for n in counts_k02]
        
        plot_name = f"position frequencies for letter {c} for {word_type_str}"
        plt.subplot(3,1,1)
        plt.title(plot_name)
        plt.bar(positions, ns_equal, label="k=0 (equal)")
        plt.legend()
        plt.subplot(3,1,2)
        plt.bar(positions, ns_k02, label="k=0.2")
        plt.legend()
        plt.subplot(3,1,3)
        plt.bar(positions, ns_freq, label="k=1 (frequency)")
        plt.legend()
        plt.gcf().set_size_inches(9,9)
        plt.savefig(f"Images/{plot_name}.png")
        plt.clf()


def get_solution_z_score(word, past_solutions, allowed_words, frequencies):
    # returns the z score of the word's nelda, which can be a rough proxy for how likely this word is to be a solution, based solely on its frequeency
    if word not in allowed_words:
        return 0
    # pretend we don't know what the solutions have been, just use them to construct a distribution across frequencies
    known_neldas = [x01_to_nelda(frequencies.get(w, 0)) for w in past_solutions]
    assert all(np.isfinite(nelda) for nelda in known_neldas), "bad nelda for known solution"
    # model as normal; based on eyeballing, it looks like it
    mu = np.mean(known_neldas)
    std = np.std(known_neldas)
    x = x01_to_nelda(frequencies.get(word, 0))
    z = (x - mu) / std
    return abs(z)


def get_guess_reward(guess, solution, reward_function):
    colors = evaluate_word(guess, solution)
    return sum(reward_function[c] for c in colors)


def get_aggregate_guess_reward(guess, solutions, reward_function):
    return np.mean([get_guess_reward(guess, solution, reward_function) for solution in solutions])


def normal_pdf(x, mu=0, sigma=1):
    z = (x - mu) / sigma
    return 1/(sigma * math.sqrt(2*np.pi)) * math.exp(-1/2 * z**2)


def create_possible_solution_set(past_solutions, allowed_words, frequencies, n_words):
    # create a simulated set of solution words based on the frequency distribution of the actual solutions
    pdf_values = []
    for w in allowed_words:
        z = get_solution_z_score(w, past_solutions, allowed_words, frequencies)
        p = normal_pdf(z)
        pdf_values.append(p)
    total_p = sum(pdf_values)
    weights = [p/total_p for p in pdf_values]
    indices = np.random.choice(list(range(len(allowed_words))), size=n_words, replace=False, p=weights)
    res = [allowed_words[i] for i in indices]

    # check its distribution
    neldas = [x01_to_nelda(frequencies.get(w, 0)) for w in res]
    past_neldas = [x01_to_nelda(frequencies.get(w, 0)) for w in past_solutions]
    xmin = min(min(neldas), min(past_neldas)) - 0.5
    xmax = max(max(neldas), max(past_neldas)) + 0.5
    plt.subplot(2,1,1)
    plt.title("neldas of simulated solutions")
    plt.gca().set_xlim((xmin, xmax))
    plt.hist(neldas, bins=50)
    plt.subplot(2,1,2)
    plt.title("neldas of actual solutions")
    plt.gca().set_xlim((xmin, xmax))
    plt.hist(past_neldas, bins=50)
    plt.show()

    return res


def find_optimal_starting_word(past_solutions, allowed_words, frequencies, reward_function):
    # want higher reward for getting a green (letter in the right place)
    # want some reward but lower for getting a yellow

    # try every allowed word as a starting point for the known solution set, see what it would have gotten you
    # but then try to bootstrap likely future solutions based on their frequencies
    rewards = {}
    for w in allowed_words:
        z = get_solution_z_score(w, past_solutions, allowed_words, frequencies)
        reward = get_aggregate_guess_reward(w, past_solutions, reward_function)
        rewards[w] = reward
        # print(f"initial guess {w} has reward {reward} and z {z}")
    for w, reward in sorted(rewards.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        print(f"initial guess {w} has average reward {reward}")
    # plt.hist(rewards.values())  # I expected most words to be relatively useless, but that's not actually true; there's a normal distribution with mean about 1.5 (with green=2, yellow=1, gray=0)
    # plt.show()


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
    past_solutions = word_history.values()
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

    # plot_frequencies_of_letters_in_positions(allowed_words, frequencies, word_type_str="allowed words")
    # plot_frequencies_of_letters_in_positions(word_history.values(), frequencies, word_type_str=f"past solutions up to puzzle #{max(word_history.keys())}")

    # test_evaluate_word()
    reward_function = {Color.RIGHT: 1.25, Color.MISPLACED: 1, Color.WRONG: 0}
    # maybe at the beginning we actually don't care that much about getting green, we'd rather just know what letters are there, so weight green and yellow the same, or green only slightly more but not double

    print("creating simulated solution set")
    new_solutions = create_possible_solution_set(past_solutions, allowed_words, frequencies, n_words=250)
    print(new_solutions)
    print("finding starting word rewards for past solutions")
    find_optimal_starting_word(past_solutions, allowed_words, frequencies, reward_function)
    print("finding starting word rewards for simulated solutions")
    find_optimal_starting_word(new_solutions, allowed_words, frequencies, reward_function)

    # query_words(allowed_words, frequencies)
