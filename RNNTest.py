import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import math


# try arranging segments so the bits in the vectors will be relatively close to features
CONSONANTS = "mnňŋptčkfsšhwryl"  # arranged into 4x4 by features
VOWELS = "ieua"  # arranged into 2x2 by features
CONSONANT_BITS = math.ceil(np.log2(len(CONSONANTS)))
VOWEL_BITS = math.ceil(np.log2(len(VOWELS)))


def get_consonant_number(c):
    return CONSONANTS.index(c)


def get_vowel_number(v):
    return VOWELS.index(v)


def get_syllable_number(cv):
    c, v = cv
    cn = get_consonant_number(c)
    vn = get_vowel_number(v)
    return cn * len(VOWELS) + vn


def get_syllable_from_number(n):
    cn, vn = divmod(n, len(VOWELS))
    return CONSONANTS[cn] + VOWELS[vn]


def get_random_syllable():
    return random.choice(CONSONANTS) + random.choice(VOWELS)


def get_random_word(n_syllables):
    return "".join(get_random_syllable() for i in range(n_syllables))


def get_training_words(n_samples):
    print("generating training sample")
    res = set()
    while len(res) < n_samples:
        n_syllables = max(1, round(np.random.normal(2.5,1)))
        w = get_random_word(n_syllables)
        res.add(w)
    print("done generating training sample")
    return list(res)


def get_words_and_classes(n_samples):
    words = get_training_words(n_samples)
    classes = []
    n_classes_possible = None
    for w in words:
        c, this_n_classes_possible = get_classification(w)
        classes.append(c)
        if n_classes_possible is None:
            n_classes_possible = this_n_classes_possible
        else:
            assert this_n_classes_possible == n_classes_possible, "mismatch n_classes_possible"
    return words, classes, n_classes_possible


def get_training_data(n_samples, padding="max"):
    words, classes, n_classes_possible = get_words_and_classes(n_samples)
    return convert_words_and_classes_to_training_data(words, classes, padding=padding, n_classes=n_classes_possible)


def convert_words_and_classes_to_training_data(words, classes, n_classes, padding="max"):
    word_vectors = [get_word_vector(x) for x in words]
    class_vectors = [get_class_vector(x, n_classes) for x in classes]
    if padding is not None:
        word_vectors = pad_word_vectors(word_vectors, padding)
    return word_vectors, class_vectors, n_classes


def pad_word_vectors(word_vectors, padding):
    if padding == "max":
        max_n_syllables = max(len(x) for x in word_vectors)
    elif type(padding) is int and padding > 0:
        max_n_syllables = padding
    else:
        raise ValueError("invalid padding value: {}".format(padding))
    placeholder_syllable = [-1 for i in range(CONSONANT_BITS + VOWEL_BITS)]
    res = []
    for vec in word_vectors:
        n_syllables = len(vec)
        n_pads_needed = max(0, max_n_syllables - n_syllables)
        placeholders = [placeholder_syllable for i in range(n_pads_needed)]
        new_vec = vec + placeholders
        res.append(new_vec)
    return res


def get_vector_from_syllable(cv):
    # I don't understand Keras Embedding and other stuff so I'm just gonna do this part myself
    c, v = cv
    cn = get_consonant_number(c)
    vn = get_vowel_number(v)
    c_vec = get_binary_vector_from_int(cn, is_vowel=False)
    v_vec = get_binary_vector_from_int(vn, is_vowel=True)
    vec = c_vec + v_vec
    assert len(vec) == CONSONANT_BITS + VOWEL_BITS
    # print(f"syllable {cv} has vector {vec}")
    return vec


def get_word_vector(w):
    syllable_vectors = []
    n_syllables = 0
    while len(w) > 0:
        cv = w[:2]
        vec = get_vector_from_syllable(cv)
        syllable_vectors.append(vec)
        n_syllables += 1
        w = w[2:]
    assert np.array(syllable_vectors).shape == (n_syllables, CONSONANT_BITS + VOWEL_BITS)
    return syllable_vectors


# make sure that the number of slots in the (one-hot-encoded) class vector matches the number of possible classes
# do this by explicitly returning the "out of N" as a second value
def get_classification(word):
    # return get_classification_unique_vowel_count(word)  # learnable 100%
    # return get_classification_custom_segments_count(word, {"a", "m", "k"})  # learnable 98%
    # return get_classification_first_and_last_consonant_are_same(word)
    return get_classification_number_of_places_of_articulation(word)


def get_class_vector(c, n_classes):
    assert c in range(n_classes), c
    lst = [int(i == c) for i in range(n_classes)]
    return lst


def get_classification_unique_vowel_count(word):
    vowels = [x for x in word if x in VOWELS]
    c = len(set(vowels))
    n_classes_possible = len(VOWELS) + 1
    return c, n_classes_possible


def get_classification_custom_segments_count(word, custom_segment_set):
    matching_segments = [x for x in word if x in custom_segment_set]
    c = len(set(matching_segments))
    n_classes_possible = len(custom_segment_set) + 1  # possible to have zero of them up to all of them
    return c, n_classes_possible


def get_classification_first_and_last_consonant_are_same(word):
    consonants = [x for x in word if x in CONSONANTS]
    same = consonants[0] == consonants[-1]
    c = 1 if same else 0
    n_classes_possible = 2
    return c, n_classes_possible


def get_classification_number_of_places_of_articulation(word):
    consonants = [x for x in word if x in CONSONANTS]
    # different ways of sorting into places will have different learnability
    places = {
        "mpfw": "labial",
        "ntsrlčš": "coronal",
        "ňy": "palatal",
        "ŋkh": "dorsal",
    }
    places_found = set()
    for cons in set(consonants):
        this_place = [v for k,v in places.items() if cons in k]
        assert len(this_place) == 1, "consonant must have a place: {}".format(cons)
        places_found.add(this_place[0])
    c = len(places_found)
    n_classes_possible = len(places) + 1  # shouldn't get 0 places but I guess it's possible
    return c, n_classes_possible


def convert_class_vectors_to_classes(class_vectors):
    classes = []
    confidences = []
    for vec in class_vectors:
        c, conf = convert_class_vector_to_class(vec)
        classes.append(c)
        confidences.append(conf)
    return classes, confidences


def convert_class_vector_to_class(class_vector):
    max_val = max(class_vector)
    c = list(class_vector).index(max_val)
    confidence = max_val / sum(class_vector)
    return c, confidence


def get_binary_vector_from_int(n, is_vowel):
    s = bin(n).replace("0b","")
    s = s.rjust(VOWEL_BITS if is_vowel else CONSONANT_BITS, "0")
    binary_lst = [int(x) for x in s]
    return binary_lst


def split_into_training_and_testing(X, Y, test_proportion):
    n_test = round(test_proportion * len(X))
    test_indices = random.sample(list(range(len(X))), n_test)
    train_indices = [i for i in range(len(X)) if i not in test_indices]
    x_train = np.array([X[i] for i in train_indices])
    x_test = np.array([X[i] for i in test_indices])
    y_train = np.array([Y[i] for i in train_indices])
    y_test = np.array([Y[i] for i in test_indices])
    return x_train, x_test, y_train, y_test


def report_accuracy(model, x_test, y_test):
    predictions = model.predict(x_test)
    prediction_classes, confidences = convert_class_vectors_to_classes(predictions)
    actual_classes, actual_confidences = convert_class_vectors_to_classes(y_test)
    assert all(x == 1 for x in actual_confidences)
    n_correct = (np.array(prediction_classes) == np.array(actual_classes)).sum()
    print(f"model got {n_correct} right out of {len(x_test)} ({100*n_correct/len(x_test)}%)")


def show_example_words_and_classes(n_samples):
    print("-- example words and classes --")
    words, classes, n_classes = get_words_and_classes(n_samples)
    for w, c in zip(words, classes):
        print(f"word {w} is in class {c}")
    input("press enter to continue")
    print("-- done with example words and classes --")


def show_example_predictions(model, n_samples, padding=None, show_raw_output_vector=False):
    words, classes, n_classes = get_words_and_classes(n_samples)
    word_vectors, class_vectors, n_classes = convert_words_and_classes_to_training_data(words, classes, padding=padding, n_classes=n_classes)
    predictions = model.predict(word_vectors)

    n_correct = 0
    predicted_classes = []
    confidences_when_correct = []
    confidences_when_wrong = []
    for w, c, w_vec, c_vec, prediction in zip(words, classes, word_vectors, class_vectors, predictions):
        prediction_class, confidence = convert_class_vector_to_class(prediction)
        predicted_classes.append(prediction_class)
        correct = prediction_class == c
        correct_str = "correct" if correct else "wrong"
        if correct:
            n_correct += 1
            confidences_when_correct.append(confidence)
        else:
            confidences_when_wrong.append(confidence)
        s = f"input word: {w} of class {c}; model predicted class {prediction_class} ({100*confidence:.2f}% confident), which is {correct_str}"
        if show_raw_output_vector:
            s += f"; raw output vector: {prediction}"
        print(s)
    print(f"model got {n_correct}/{n_samples} correct ({100*n_correct/n_samples}%)")
    show_confusion_matrix(predicted_classes, classes)
    show_confidence_plot(confidences_when_correct, confidences_when_wrong)


def show_confusion_matrix(predicted, observed):
    predicted_classes = set(predicted)
    observed_classes = set(observed)
    all_classes = sorted(predicted_classes | observed_classes)
    d = {(i,j):0 for i in all_classes for j in all_classes}
    for pred, obs in zip(predicted, observed):
        d[(pred,obs)] += 1

    df = pd.DataFrame(index=all_classes)
    df.index.name = "predicted \\ observed"
    for observed_class in all_classes:
        colname = str(observed_class)
        series = [d[(predicted_class, observed_class)] for predicted_class in all_classes]
        df[colname] = series
    print("\nconfusion matrix:")
    print(df)


def show_confidence_plot(conf_correct, conf_wrong):
    all_confidence_xs = np.linspace(0, 1, 1000)
    add_kde_to_plot(conf_correct, all_confidence_xs, c="b", label="correct prediction")
    add_kde_to_plot(conf_wrong, all_confidence_xs, c="r", label="incorrect prediction")
    plt.title("confidence levels")
    plt.show()


def add_kde_to_plot(x, all_xs, **kwargs):
    kde = stats.gaussian_kde(x)
    # plt.hist(x, density=True, bins=100, alpha=0.3)
    plt.plot(all_xs, kde(all_xs), **kwargs)
    


if __name__ == "__main__":
    show_example_words_and_classes(n_samples=100)

    input_dim = CONSONANT_BITS + VOWEL_BITS  # length of vector at each time step
    timesteps_per_input = None  # variable length input sequences
    input_shape = (timesteps_per_input, input_dim)

    X, Y, n_classes_possible = get_training_data(40000, padding="max")
    output_len = n_classes_possible

    input_layer = layers.Input(input_shape)
    simple_rnn = layers.SimpleRNN(128, activation="relu")
    output_layer = layers.Dense(output_len, activation="sigmoid")

    model = keras.Sequential()
    model.add(input_layer)  # add one-by-one for debugging purposes
    model.add(simple_rnn)
    model.add(output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error")

    padded_length = len(X[0])

    test_proportion = 0.1
    x_train, x_test, y_train, y_test = split_into_training_and_testing(X, Y, test_proportion=test_proportion)

    # reserve some training data as validation data during training epochs, to check for overfit (not the same as testing data)
    validation_proportion = 0.1
    n_validation_samples = round(len(x_train) * validation_proportion)
    x_val = x_train[-n_validation_samples:]
    y_val = y_train[-n_validation_samples:]
    x_train = x_train[:-n_validation_samples]
    y_train = y_train[:-n_validation_samples]

    # for when the data is padded to same length per sample (but I fear this is skewing the results because the reported accuracy on the test data does not match the accuracy measured on randomly generated new data)
    epochs = 3
    batch_size = 50
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_val, y_val))
    
    report_accuracy(model, x_test, y_test)

    show_raw_output_vector = False
    show_example_predictions(model, n_samples=1000, padding=padded_length, show_raw_output_vector=show_raw_output_vector)

