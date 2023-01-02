import random
import numpy as np
import matplotlib.pyplot as plt

import FitWord2Vec as w2v


articulator_state_to_number = {
    "blade_apical": {
        "default": 0,
        "approximant": 0.3,
        "fricative": 0.7,
        "tap": 0.9,
        "stop": 1,
    },
    "blade_laminal": {
        "default": 0,
        "approximant": 0.3,
        "fricative": 0.7,
        "stop": 1,
    },
    "blade_horizontal": {
        "default": 0,
        "interdental": 1,
        "dental": 0.5,
        "alveolar": -0.25,
        "postalveolar": -0.5,
        "retroflex": -1,
    },
    "dorsum_horizontal": {
        "uvular": -1,
        "velar": -0.75,
        "back": -0.5,
        "default": 0,
        "central": 0.5,
        "palatal": 1,
        "front": 1,
    },
    "dorsum_vertical": {
        "default": 0,
        "approximant": 0.3,
        "fricative": 0.7,
        "stop": 1,
    },
    "tongue_sides": {
        "default": 0,
        "approximant": 0.3,
        "fricative": 0.7,
        "tap": 0.9,
        "stop": 1,
    },
    "jaw": {
        "low": -1,
        "default": 0,
        "mid": 0,
        "high": 1,
    },
    "lips_vertical": {
        "default": 0,
        "compressed": 0.5,
        "closed": 1,
    },
    "lips_horizontal": {
        "default": 0,
        "rounded": 1,
    },
    "glottis": {
        "voiceless": -1,
        "breathy": -0.5,
        "default": 0,
        "voiced": 0.3,
        "creaky": 0.7,
        "closed": 1,
    },
    "nasal": {
        "default": 0,
        "open": 1,
    },
    "teeth": {
        "default": 0,
        "closed": 1,
    }
}
articulators = sorted(articulator_state_to_number.keys())
N_FEATURES = len(articulators)

features_by_sound = {
    "a": {
        "jaw": "low",
        "glottis": "voiced",
    },
    "i": {
        "jaw": "high",
        "dorsum_horizontal": "front",
        "glottis": "voiced",
    },
    "u": {
        "jaw": "high",
        "dorsum_horizontal": "back",
        "glottis": "voiced",
        "lips_horizontal": "rounded",
    },
    "o": {
        "jaw": "mid",
        "dorsum_horizontal": "back",
        "glottis": "voiced",
        "lips_horizontal": "rounded",
    },
    "e": {
        "jaw": "mid",
        "dorsum_horizontal": "front",
        "glottis": "voiced",
    },
    "y": {  # change to <j> once start doing more IPA
        "jaw": "high",
        "dorsum_vertical": "approximant",
        "dorsum_horizontal": "front",
        "glottis": "voiced",
    },
    "w": {
        "jaw": "high",
        "dorsum_horizontal": "back",
        "dorsum_vertical": "approximant",
        "glottis": "voiced",
        "lips_horizontal": "rounded",
    },
    "k": {
        "dorsum_horizontal": "velar",
        "dorsum_vertical": "stop",
        "glottis": "voiceless",
    },
    "t": {
        "blade_apical": "stop",
        "blade_horizontal": "alveolar",
        "glottis": "voiceless",
    },
    "p": {
        "lips_vertical": "closed",
        "glottis": "voiceless",
    },
    "n": {
        "nasal": "open",
        "blade_apical": "stop",
        "blade_horizontal": "alveolar",
        "glottis": "voiced",
    },
    "m": {
        "nasal": "open",
        "lips_vertical": "closed",
        "glottis": "voiced",
    },
    "s": {
        "teeth": "closed",
        "glottis": "voiceless",
        "blade_apical": "fricative",
        "blade_horizontal": "alveolar",
    },
    "r": {
        "blade_apical": "tap",
        "blade_horizontal": "alveolar",
        "glottis": "voiced",
    },
    "l": {
        "blade_apical": "stop",  # it touches completely since the air comes around the sides
        "blade_horizontal": "alveolar",
        "tongue_sides": "approximant",
        "glottis": "voiced",
    }
}

TIME_LENGTH = 120  # needs to be constant per input



def convert_word_to_features(w):
    segs = list(w)  # later can do something more complicated
    return [features_by_sound[s] for s in segs]


def convert_feature_dict_to_vector(d):
    res = []
    for a in articulators:
        if a in d:
            val_str = d[a]
            val = articulator_state_to_number[a][val_str]
        else:
            val = articulator_state_to_number[a]["default"]
        res.append(val)
    return res


def convert_word_to_articulatory_array(w):
    seg_features = convert_word_to_features(w)
    n_segs = len(seg_features)
    if len(seg_features) >= TIME_LENGTH:
        raise ValueError("word too long")
    arr = np.zeros((TIME_LENGTH, N_FEATURES))
    # place the segments approximately evenly divided along the time axis
    for i in range(n_segs):
        start_frac = i / n_segs
        end_frac = (i + 1) / n_segs
        start_index = round(start_frac * TIME_LENGTH)
        end_index = round(end_frac * TIME_LENGTH)
        assert start_index < end_index, "must fill at least one time unit; word may be too long"
        if i == n_segs - 1:
            assert end_index == TIME_LENGTH, "didn't fill whole time axis"
        d = seg_features[i]
        vec = convert_feature_dict_to_vector(d)
        for j in range(start_index, end_index):
            # not sure I trust doing like arr[start_index:end_index, :]
            # in case slice is square, don't want transposed vector
            arr[j, :] = vec
    return arr


def plot_words(words):
    for w in words:
        arr = convert_word_to_articulatory_array(w)
        print(w)
        row_sums = np.sum(abs(arr), axis=1)
        assert (row_sums != 0).all()
        plt.imshow(arr.T)
        plt.gcf().set_size_inches(12, 3)
        plt.title(w)
        plt.savefig(f"images/{w}.png")
        plt.gcf().clear()


def convert_word_to_nn_input(w):
    arr = convert_word_to_articulatory_array(w)
    vec = arr.flatten()
    # TODO play with different input flattening (row-vs-column major) and see if it affects the results
    assert vec.shape == (N_FEATURES * TIME_LENGTH,), vec.shape
    return vec


def convert_words_to_articulatory_nn_input(words):
    n_words = len(words)
    n_cols = N_FEATURES * TIME_LENGTH
    arr = np.zeros((n_words, n_cols))
    for i, w in enumerate(words):
        vec = convert_word_to_nn_input(w)
        arr[i,:] = vec
    return arr


if __name__ == "__main__":
    # if use convolution, do it along time axis only
    # don't bleed information across the other axis
    # because articulators should be treated as independent (for now)

    # all_words = w2v.get_all_words_from_text_tokens(text_tokens)
    # plot_words(all_words)

    pass