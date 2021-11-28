# generate languages with weirdly specific joint distributions of bigrams (Markov table is sparse)
# assume equal transition probability to each of the options
# assume CV alternate but word can start and end with anything

import random

CONSONANTS = [
    "m", "n",
    "p", "t", "ts", "č", "k", "q", "'",
    "b", "d", "dz", "ǰ", "g",
    "mb", "nd", "ndz", "nǰ", "ŋg",
    "p'", "t'", "ts'", "č'", "k'", "q'",
    "ph", "th", "tsh", "čh", "kh", "qh",
    "f", "s", "š", "h",
    "w", "l", "r", "j",
]
VOWELS = [
    "a", "o", "e", "u", "i",
    "aa", "oo", "ee", "uu", "ii",
]


def get_random_transition_possibilities(min_p=0, max_p=0.5):
    d = {}
    for c in CONSONANTS:
        p = random.uniform(min_p, max_p)
        options = random_nonempty_subset(VOWELS, p)
        d[c] = options
    for v in VOWELS:
        p = random.uniform(min_p, max_p)
        options = random_nonempty_subset(CONSONANTS, p)
        d[v] = options
    # add word boundaries
    for k in d:
        d[k].append("#")
    d["#"] = list(CONSONANTS + VOWELS)
    return d


def get_unrestricted_transition_possibilities():
    return get_random_transition_possibilities(min_p=1, max_p=1)


def random_nonempty_subset(lst, probability_per_option):
    assert 0 < probability_per_option <= 1
    while True:
        res = []
        for x in lst:
            if random.random() < probability_per_option:
                res.append(x)
        if len(res) > 0:
            return res


def get_word(transitions, min_length=2):
    w = []
    while True:
        last = w[-1] if len(w) > 0 else "#"
        options = transitions[last]
        chosen = random.choice(options)
        if chosen == "#":
            if len(w) >= min_length:  # don't do and because then you'll get # in middle of word (when chosen is # but word isn't long enough yet, it'll go to the else)
                break
        else:
            w.append(chosen)
    return "".join(w)


def get_words(transitions, n):
    return [get_word(transitions) for i in range(n)]


def get_fake_word(transitions):
    # do something like at each step, with some probability, make an illegal transition; want the lengths to look similar to the real words
    raise NotImplementedError


def train_on_real_words(transitions):
    print("Here are some real words in the language. Press enter after each one to continue.")
    for i in range(100):
        input(get_word(transitions))
    print("Done with training!")


if __name__ == "__main__":
    transitions = get_random_transition_possibilities()
    # unrestricted_transitions = get_unrestricted_transition_possibilities()
    train_on_real_words(transitions)
