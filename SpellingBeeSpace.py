# try to quantify distances between the words in a game of Spelling Bee (New York Times)
# see how well my intuition that some words are "far away" from others holds up
# e.g. "imam" is often possible but not easily reachable by connections to other words, whereas a set like "hood", "hooded", "hoed" feels much more connected by mutual similarity
# probably can't do morphology in this program, just do string stuff

import random
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

from Levenshtein import levenshtein, normalized_levenshtein


def get_words():
    with open("Language/cmudict.txt") as f:
        lines = f.readlines()
    words = []
    for l in lines:
        w, *_ = l.strip().split(" ")
        if all(x in string.ascii_uppercase for x in w):
            words.append(w)
    return list(set(words))


def get_pangram(words):
    options = [w for w in words if len(set(w)) == 7]
    return random.choice(options)


def get_words_with_letters(words, letters):
    return list(set([w for w in words if len(w) >= 4 and set(w) - letters == set()]))


def get_multidimensional_distance_matrix(words):
    # every cell has an object with all the different kinds of distances
    m = [[None for j in range(len(words))] for i in range(len(words))]
    for i in range(len(words)):
        for j in range(i, len(words)):
            dists = get_multidimensional_distances(words[i], words[j])
            m[i][j] = dists
            m[j][i] = dists
    return m


def get_distance_matrix(words):
    m0 = get_multidimensional_distance_matrix(words)
    m = [[None for j in range(len(words))] for i in range(len(words))]
    for i in range(len(words)):
        for j in range(i, len(words)):
            dists = m0[i][j]
            assert m0[j][i] == dists, "non-symmetrical input matrix"
            d = get_distance_from_multidimensional_distances(dists)
            m[i][j] = d
            m[j][i] = d
    m = np.array(m)
    return m


def substring_distance(w1, w2):
    # if neither is substring of the other, return 1
    # otherwise, what proportion is the shorter of the longer?
    # remember this is a DISTANCE measure so 0 means equal
    if len(w1) == len(w2):
        return 0 if w1 == w2 else 1
    elif len(w1) < len(w2):
        shorter, longer = w1, w2
    else:
        shorter, longer = w2, w1
    if shorter in longer:
        return 1 - len(shorter) / len(longer)
    else:
        return 1


def get_multidimensional_distances(w1, w2):
    # various kinds of measurements of distances between the words
    lev = levenshtein(w1, w2)
    nlev = normalized_levenshtein(w1, w2)
    subs = substring_distance(w1, w2)

    return {
        "levenshtein": lev,
        "normalized_levenshtein": nlev,
        "substring": subs,
    }


def get_distance_from_multidimensional_distances(dists):
    # here we can do weighting stuff, adjusting for correlation/collinearity, etc.
    weights = {
        "levenshtein": 0,
        "normalized_levenshtein": 0.5,
        "substring": 2,
    }
    scalars = []
    for k, v in dists.items():
        weight = weights.get(k, 1)
        scalars.append(weight * v)
    norm_func = np.linalg.norm
    return norm_func(scalars)



if __name__ == "__main__":
    words = get_words()
    pangram = get_pangram(words)
    letters = set(pangram)
    answers = get_words_with_letters(words, letters)
    for x in answers:
        print(x)
    print(letters)
    m = get_distance_matrix(answers)

    mds_fit = MDS(dissimilarity="precomputed").fit_transform(m)
    xs, ys = mds_fit.T
    plt.scatter(xs, ys)
    for i in range(len(answers)):
        plt.gca().annotate(answers[i], (xs[i], ys[i]))
    plt.show()
