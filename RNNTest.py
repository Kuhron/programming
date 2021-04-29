import numpy as np
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


def get_classification(word):
    vowels = [x for x in word if x in VOWELS]
    c = len(set(vowels))
    # print(f"{word} is in class {c}")
    return c


def get_words_and_classes(n_samples):
    words = get_training_words(n_samples)
    classes = [get_classification(w) for w in words]
    return words, classes


def get_training_data(n_samples, pad=True):
    words, classes = get_words_and_classes(n_samples)
    return convert_words_and_classes_to_training_data(words, classes, pad=pad)


def convert_words_and_classes_to_training_data(words, classes, pad=True):
    word_vectors = [get_word_vector(x) for x in words]
    class_vectors = [get_class_vector(x) for x in classes]
    if pad:
        word_vectors = pad_word_vectors(word_vectors)
    return word_vectors, class_vectors


def pad_word_vectors(word_vectors):
    max_n_syllables = max(len(x) for x in word_vectors)
    placeholder_syllable = [-1 for i in range(CONSONANT_BITS + VOWEL_BITS)]
    res = []
    for vec in word_vectors:
        n_syllables = len(vec)
        placeholders = [placeholder_syllable for i in range(max_n_syllables - n_syllables)]
        new_vec = vec + placeholders
        res.append(new_vec)
    return res


def get_batches_by_length(x_train, y_train):
    # for variable-length data
    # return them in random order
    print("getting batches by length")
    res = []
    indices_not_used = list(range(len(x_train)))
    while len(indices_not_used) > 0:
        chosen_sample_i = random.choice(indices_not_used)
        chosen_x_len = len(x_train[chosen_sample_i])
        x_train_subset_with_indices = [(i,x) for i, x in enumerate(x_train) if len(x) == chosen_x_len]
        subset_indices = [i for i,x in x_train_subset_with_indices]
        x_train_subset = [x for i,x in x_train_subset_with_indices]
        y_train_subset = [y_train[i] for i in subset_indices]
        indices_not_used = [i for i in indices_not_used if i not in subset_indices]
        try:
            arr = np.array(x_train_subset)
        except:
            raise  # I wrote it this way to indicate that I expect errors may occur but don't yet know what
        res.append([chosen_x_len, x_train_subset, y_train_subset])
    print("done getting batches by length")
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


def get_class_vector(c):
    assert c in range(len(VOWELS)+1), c
    lst = [int(i == c) for i in range(len(VOWELS)+1)]
    return lst


def convert_class_vectors_to_classes(class_vectors):
    return [convert_class_vector_to_class(vec) for vec in class_vectors]


def convert_class_vector_to_class(class_vector):
    return list(class_vector).index(max(class_vector))


def get_binary_vector_from_int(n, is_vowel):
    s = bin(n).replace("0b","")
    s = s.rjust(VOWEL_BITS if is_vowel else CONSONANT_BITS, "0")
    binary_lst = [int(x) for x in s]
    return binary_lst


def split_into_training_and_testing(X, Y, test_proportion):
    print("splitting into training and testing data")
    n_test = round(test_proportion * len(X))
    test_indices = random.sample(list(range(len(X))), n_test)
    train_indices = [i for i in range(len(X)) if i not in test_indices]
    x_train = [X[i] for i in train_indices]
    x_test = [X[i] for i in test_indices]
    y_train = [Y[i] for i in train_indices]
    y_test = [Y[i] for i in test_indices]
    print("done splitting into training and testing data")
    return x_train, x_test, y_train, y_test


def report_accuracy(model, x_test, y_test):
    predictions = model.predict(x_test)
    prediction_classes = convert_class_vectors_to_classes(predictions)
    actual_classes = convert_class_vectors_to_classes(y_test)
    n_correct = (np.array(prediction_classes) == np.array(actual_classes)).sum()
    print(f"model got {n_correct} right out of {len(x_test)} ({100*n_correct/len(x_test)}%)")


def show_example_predictions(model, n_samples):
    words, classes = get_words_and_classes(n_samples)
    word_vectors, class_vectors = convert_words_and_classes_to_training_data(words, classes)
    predictions = model.predict(word_vectors)

    n_correct = 0
    for w, c, w_vec, c_vec, prediction in zip(words, classes, word_vectors, class_vectors, predictions):
        prediction_class = convert_class_vector_to_class(prediction)
        if prediction_class == c:
            correct = "correct"
            n_correct += 1
        else:
            correct = "wrong"
        print(f"input word: {w} of class {c}; model predicted class {prediction_class}, which is {correct}")
    print(f"model got {n_correct}/{n_samples} correct ({100*n_correct/n_samples}%)")


def fit_model_homebrew_length_batching(x_train, y_train, model):
    # homebrew so it will look at samples without padding, do each length as a separate batch or set of batches
    batches_by_length = get_batches_by_length(x_train, y_train)
    for epoch in range(200):
        print("homebrew epoch", epoch)
        random.shuffle(batches_by_length)  # look at them in different order every time
        for sample_length, x_train_subset, y_train_subset in batches_by_length:
            model.fit(x_train_subset, y_train_subset, batch_size=50, shuffle=True)
    print("done training homebrew")


if __name__ == "__main__":
    input_dim = CONSONANT_BITS + VOWEL_BITS  # length of vector at each time step
    timesteps_per_input = None  # variable length input sequences
    input_shape = (timesteps_per_input, input_dim)
    n_classes = len(VOWELS)+1  # possible numbers of vowels
    output_len = n_classes

    input_layer = layers.Input(input_shape)
    simple_rnn = layers.SimpleRNN(128, activation="relu")
    output_layer = layers.Dense(output_len, activation="sigmoid")

    model = keras.Sequential()
    model.add(input_layer)  # add one-by-one for debugging purposes
    model.add(simple_rnn)
    model.add(output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error")

    X, Y = get_training_data(100000, pad=False)

    x_train, x_test, y_train, y_test = split_into_training_and_testing(X, Y, test_proportion=0.1)

    # reserve some training data as validation data during training epochs, to check for overfit (not the same as testing data)
    # n_validation_samples = round(len(x_train) * 0.1)
    # x_val = x_train[-n_validation_samples:]
    # y_val = y_train[-n_validation_samples:]
    # x_train = x_train[:-n_validation_samples]
    # y_train = y_train[:-n_validation_samples]

    # for when the data is padded to same length per sample (but I fear this is skewing the results because the reported accuracy on the test data does not match the accuracy measured on randomly generated new data)
    # model.fit(x_train, y_train, epochs=20, batch_size=50, shuffle=True, validation_data=(x_val, y_val))


    fit_model_homebrew_length_batching(x_train, y_train, model)
    
    # report_accuracy(model, x_test, y_test)  # better for when data is padded for constant sample length

    show_example_predictions(model, n_samples=1000)
