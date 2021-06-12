# implementation of the neural net agents who learn to communicate with the articulations/spectra in BasisSpectra.py

import keras
from keras import layers
from keras.utils.np_utils import to_categorical
# from keras.datasets import mnist
# import BasisSpectra as bs
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import itertools
import string
# import Music.WavUtil as wav


# agent babbles in order to learn what its articulations sound like, so it can train its ear
# the Mouth is a device endowed upon the agent, cannot learn anything, implemented in BasisSpectra.py with Articulator classes
# don't bother with sound waves, just give them spectra directly, can add noise to them but don't bother with fft and ifft
# the Ear is a neural net from spectrum to articulation vector
# let articulation vector be the internal representation
# the Eye is a neural net from image to internal representation (articulation vector)
# and the Interpreter is a neural net from internal representation to image

# babbling task: agent creates random articulation vector, resulting spectrum (sound, with noise) is fed to the Ear, the Ear's predicted articulation vector is evaluated relative to the actual one
# multi-agent picture description task: agents take turns, in each turn, one agent (the Describer) sees image, the Eye creates internal representation from it, the Mouth says it, the resulting spectrum (with noise) is fed to all agents' (including the Describer's) Ears, they create articulation vectors using the Ear's prediction, then the Interpreter (again for all agents including the Describer) creates an image from the articulation vector which the Ear output, this image is then evaluated versus the actual one, train the Interpreter and Ear this way
# how is the Eye trained? all listeners (including the Describer itself) should take the articulatory representation spoken for that picture, use that and the actual image to train the Eye; the Eye might have to be initialized very roughly by associating some random articulation vectors with random images, and then the agents can work together from there, just so it can output something for the first turn when no words have been learned yet

# simpler version as of 2021-05-28:
# instead of images, they receive a very simple input which is a number from 0 to 1, they should create classification along this scale
# instead of articulation stuff, they output a vector of say 5 values from 0 to 1, and this is rounded to 0 or 1 for each value
# this is where anatomical differences can be introduced, where is each one's threshold for outputting 0 or 1 (endowed e.g. one agent has thresholds [0.49, 0.51, 0.5, 0.5, 0.47], another has [0.4, 0.41, 0.61, 0.55. 0.57])


class SimpleAgent:
    def __init__(self, name, n_articulation_positions, bias_vector=None, noise_stdev=None):
        if bias_vector is not None:
            raise Exception("bias is deprecated for the purposes of projects in Spring 2021")
        self.name = name
        self.bias_vector = bias_vector
        self.noise_stdev = noise_stdev

        self.n_articulation_positions = n_articulation_positions
        self.receptors_per_articulator = n_articulation_positions  # can change this to group some articulations together
        self.production_model = SimpleAgent.get_production_model(self.receptors_per_articulator, self.n_articulation_positions)
        self.perception_model = SimpleAgent.get_perception_model(self.receptors_per_articulator)

    def __repr__(self):
        return f"<SimpleAgent {self.name}>"

    @staticmethod
    def random(name, n_articulation_positions, bias_stdev=None, noise_stdev=None):
        if bias_stdev is not None:
            raise Exception("bias is deprecated for the purposes of projects in Spring 2021")
        print("getting random SimpleAgent")
        # output_vector_len = 5
        # bias_vector = np.random.normal(0, bias_stdev, (output_vector_len,))
        a = SimpleAgent(name, n_articulation_positions, noise_stdev=noise_stdev)
        print("done getting SimpleAgent")
        return a

    @staticmethod
    def get_production_model(receptors_per_articulator, n_articulation_positions):
        n_articulators = 5
        output_layer_len = n_articulators * n_articulation_positions  # do this for simplicity to try to get rid of the iconicity of gradability, the perception and production of different positions of an articulator are treated as different targets, not along a scale
        hidden_layer_len = 50
        input_layer_len = 1

        input_layer = layers.InputLayer(input_layer_len)
        hidden_layer = layers.Dense(hidden_layer_len, activation="relu")
        # hidden_layer2 = layers.Dense(hidden_layer_len, activation="relu")
        output_layer = layers.Dense(output_layer_len, activation="sigmoid")

        model = keras.Sequential()
        model.add(input_layer)
        model.add(hidden_layer)
        # model.add(hidden_layer2)
        model.add(output_layer)
        opt = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(opt, loss="mean_squared_error")

        return model

    @staticmethod
    def get_perception_model(receptors_per_articulator):
        n_articulators = 5
        input_layer_len = n_articulators * receptors_per_articulator
        hidden_layer_len = 50
        output_layer_len = 1

        input_layer = layers.InputLayer(input_layer_len)
        hidden_layer = layers.Dense(hidden_layer_len, activation="relu")
        # hidden_layer2 = layers.Dense(hidden_layer_len, activation="relu")
        output_layer = layers.Dense(output_layer_len, activation="sigmoid")

        model = keras.Sequential()
        model.add(input_layer)
        model.add(hidden_layer)
        # model.add(hidden_layer2)
        model.add(output_layer)
        opt = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(opt, loss="mean_squared_error")

        return model

    def convert_output_to_pronunciation(self, output):
        bias = 0  #self.bias_vector
        noise = np.random.normal(0, self.noise_stdev, (5 * self.n_articulation_positions,))
        res = output + bias + noise
        res = np.maximum(0, np.minimum(1, res))
        # res = round_array_to_n_ticks_01(res, n_ticks=n_chars)
        # assert (0 <= res).all() and (res <= 1).all()  # don't want to train on output that's outside this range
        pronunciation_one_hot = get_receptor_category_one_hot_stacked_from_01(res, self.receptors_per_articulator)
        return pronunciation_one_hot

    def describe(self, inp):
        if len(inp.shape) == 1:
            n_words, = inp.shape
        elif len(inp.shape) == 2:
            n_words, one = inp.shape
            assert one == 1
        else:
            raise Exception("bad shape {inp.shape}")

        outp = self.production_model.predict(inp)
        assert outp.shape == (n_words, 5 * self.n_articulation_positions)
        pronunciation_01 = outp

        # old way
        # pronunciation_01 = self.convert_output_to_pronunciation(outp)
        assert (0 <= pronunciation_01).all() and (pronunciation_01 <= 1).all()
        # print(f"Agent {self.name} described\n{inp}\nas\n{outp}\npronounced as\n{pronunciation}")
        # print(f"{inp} -> {self.name} -> {pronunciation}")
        return pronunciation_01

    def describe_as_string(self, meaning):
        n_chars = self.n_articulation_positions
        pronunciation_one_hot = self.describe(meaning)
        chars = string.ascii_uppercase
        one, pronunciation_one_hot_len = pronunciation_one_hot.shape
        assert one == 1
        assert pronunciation_one_hot_len == 5 * n_chars
        arr = pronunciation_one_hot.reshape(5, n_chars)
        char_indices = np.argmax(arr, axis=-1)
        s = "".join(chars[i] for i in char_indices)
        return s

    def perceive(self, pronunciation, meaning, epochs=50, verbose=0):
        n_receptors = self.receptors_per_articulator

        # old way
        # pronunciation_input_vector = get_receptor_category_one_hot_stacked(pronunciation, n_receptors)

        pronunciation_input_vector = pronunciation
        assert pronunciation_input_vector.shape == (pronunciation.shape[0], 5 * self.n_articulation_positions), pronunciation_input_vector.shape

        # predicted_meaning = self.perception_model.predict(pronunciation_input_vector)
        # self_pronunciation_of_predicted_meaning = self.describe(predicted_meaning, chars)
        # pronunciations_are_same = pronunciation == self_pronunciation_of_predicted_meaning
        # category_similarity = pronunciations_are_same.mean()
        # print(f"{self.name} has category similarity of {category_similarity} to the describer")
        # diff = predicted_meaning - meaning
        # avg_error = (diff**2).mean()
        # print(f"Agent {self.name} heard\n{pronunciation}\nand interpreted it as meaning\n{predicted_meaning}")
        # print(f"{pronunciation} -> {self.name} -> {predicted_meaning} (diff {predicted_meaning-meaning})")
        # print(f"{self.name} understood with mean squared error {avg_error}")

        self.perception_model.fit(pronunciation_input_vector, meaning, verbose=verbose, epochs=epochs)
        self.production_model.fit(meaning, pronunciation_input_vector, verbose=verbose, epochs=epochs)

    def seed(self, n_samples, epochs, condition):
        print(f"seeding {self.name}")
        pronunciations, meanings = get_predetermined_categorization_seeding_data(n_samples, condition, self.n_articulation_positions, self.receptors_per_articulator)
        self.perceive(pronunciations, meanings, epochs)

    def get_pronunciations_of_meanings(self):
        meanings = np.linspace(0, 1, 26)  # this 26 is not about articulation positions, it's just so I have a nice list of decimal meanings
        res = []
        for m in meanings:
            inp = np.array([m]).reshape(1,1)
            s = self.describe_as_string(inp)
            res.append((m,s))
        return res

    def get_language_vector(self):
        # a numerical array which will allow for direct comparison of the languages of different agents
        meanings = np.linspace(0, 1, 26)
        pronunciations_01 = self.describe(meanings)
        return pronunciations_01

    def report_pronunciations_of_meanings(self):
        tups = self.get_pronunciations_of_meanings()
        for m, s in tups:
            print(f"{self.name} describes {m} as {s}")

    def report_meanings_of_pronunciations(self):
        vector_len = 5
        bits = [0,1]
        possibilities = [bits] * vector_len
        cartesian = list(itertools.product(*possibilities))
        assert len(cartesian) == 32
        for vec in sorted(cartesian):
            pronunciation = np.array(vec).reshape(1, vector_len)
            meaning = self.perception_model.predict(pronunciation)
            print(f"{self.name} thinks {pronunciation} means {meaning}")


class BasisSpectrumAgent:
    def __init__(self, name, articulators, image_vector_len, n_articulation_positions_per_sequence, noise_average_amplitude):
        self.name = name
        self.articulators = articulators
        self.image_vector_len = image_vector_len
        self.n_articulation_positions_per_sequence = n_articulation_positions_per_sequence
        self.noise_average_amplitude = noise_average_amplitude

        self.mouth = Mouth(articulators, n_articulation_positions_per_sequence=n_articulation_positions_per_sequence, noise_average_amplitude=noise_average_amplitude)

        self.single_articulation_vector_len = self.mouth.single_artv_len
        self.full_articulation_vector_len = self.mouth.full_artv_len
        self.single_spectrum_vector_len = self.mouth.single_specv_len
        self.full_spectrum_vector_len = self.mouth.full_specv_len

        eye_input_layer = keras.Input(shape=(self.image_vector_len,))
        eye_hl0 = layers.Dense(self.image_vector_len, activation="relu")(eye_input_layer)
        eye_hl1 = layers.Dense(self.image_vector_len, activation="relu")(eye_hl0)
        eye_hidden_layers = [eye_hl0, eye_hl1]
        # if the articulators expect articulator param values in [0, 1], then anything outputting articulation vector should have activation of sigmoid
        eye_output_regularizer = keras.regularizers.l2(l2=1e-2)  # penalize large values of articulation vector components
        eye_output_layer = layers.Dense(self.full_articulation_vector_len, activation="sigmoid", 
            activity_regularizer=eye_output_regularizer)(eye_hidden_layers[-1])
        self.eye = Eye(eye_input_layer, eye_hidden_layers, eye_output_layer)

        ear_input_layer = keras.Input(shape=(self.full_spectrum_vector_len,))
        ear_hl0 = layers.Dense(self.full_spectrum_vector_len, activation="relu")(ear_input_layer)
        ear_hl1 = layers.Dense(self.full_spectrum_vector_len, activation="relu")(ear_hl0)
        ear_hidden_layers = [ear_hl0, ear_hl1]
        ear_output_layer = layers.Dense(self.full_articulation_vector_len, activation="sigmoid")(ear_hidden_layers[-1])
        self.ear = Ear(ear_input_layer, ear_hidden_layers, ear_output_layer)

        ip_input_layer = keras.Input(shape=(self.full_articulation_vector_len,))
        ip_hl0 = layers.Dense(self.full_articulation_vector_len, activation="relu")(ip_input_layer)
        ip_hl1 = layers.Dense(self.full_articulation_vector_len, activation="relu")(ip_hl0)
        ip_hidden_layers = [ip_hl0, ip_hl1]
        ip_output_layer = layers.Dense(self.image_vector_len)(ip_hidden_layers[-1])
        self.interpreter = Interpreter(ip_input_layer, ip_hidden_layers, ip_output_layer)

    def __repr__(self):
        return "<Agent {}>".format(self.name)

    def babble(self, n_samples, epochs, batch_size):
        print("\n-- babbling {}".format(self.name))
        x_train, y_train = self.create_babble_dataset_for_ear(n_samples)
        expected_x_train_shape = (n_samples, self.full_spectrum_vector_len)
        expected_y_train_shape = (n_samples, self.full_articulation_vector_len)
        assert x_train.shape == expected_x_train_shape, "expected {}, got {}".format(expected_x_train_shape, x_train.shape)
        assert y_train.shape == expected_y_train_shape, "expected {}, got {}".format(expected_y_train_shape, y_train.shape)
        x_test, y_test = self.create_babble_dataset_for_ear(max(10, int(n_samples*0.1)))
        self.ear.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))
        print("\n-- done babbling {}".format(self.name))

    def create_babble_dataset_for_ear(self, n_samples):
        x = []
        y = []
        for i in range(n_samples):
            this_x, this_y = self.create_single_babble_data_point_for_ear()
            assert len(this_x) == self.full_spectrum_vector_len, "expected len {}, got {}".format(self.full_spectrum_vector_len, len(this_x))
            assert len(this_y) == self.full_articulation_vector_len, "expected len {}, got {}".format(self.full_articulation_vector_len, len(this_y))
            x.append(this_x)
            y.append(this_y)
        return np.array(x), np.array(y)

    def create_single_babble_data_point_for_ear(self):
        articulation_vector = self.mouth.get_random_full_articulation_vector(dimensions=1)
        spectrum = self.mouth.pronounce(articulation_vector)
        assert len(articulation_vector) == self.full_articulation_vector_len, "expected len {}, got {}".format(self.full_articulation_vector_len, len(articulation_vector))
        assert len(spectrum) == self.full_spectrum_vector_len, "expected len {}, got {}".format(self.full_spectrum_vector_len, len(spectrum))
        
        # debug
        # articulations = self.mouth.convert_full_articulation_vector_to_articulation_sequence(articulation_vector)
        # spectra = self.mouth.convert_full_spectrum_vector_to_spectrum_sequence(spectrum)
        # show_articulations_and_spectra_simple(articulations, spectra)

        return (spectrum, articulation_vector)

    def get_random_seed_articulation_vectors_for_images(self, images):
        n_images, *single_image_shape = images.shape
        image_vector_len = np.prod(single_image_shape)
        assert image_vector_len == self.image_vector_len
        articulation_vectors = [self.mouth.get_random_full_articulation_vector(dimensions=1) for i in range(n_images)]
        articulation_vectors = np.array(articulation_vectors)
        return articulation_vectors

    def seed_eye(self, images, epochs):
        print("\n-- seeding eye {}".format(self.name))
        # give the Eye just a few images along with random articulations to train on, so it's not starting from nothing
        articulation_vectors = self.get_random_seed_articulation_vectors_for_images(images)
        images = images.reshape(images.shape[0], self.image_vector_len)
        self.eye.model.fit(images, articulation_vectors, epochs=epochs, shuffle=True)
        print("\n-- done seeding eye {}".format(self.name))

    def seed_interpreter(self, images, epochs):
        print("\n-- seeding interpreter {}".format(self.name))
        # give the interpreter a few random articulation vectors and images to train on, so it's not starting from nothing
        articulation_vectors = self.get_random_seed_articulation_vectors_for_images(images)
        images = images.reshape(images.shape[0], self.image_vector_len)
        self.interpreter.model.fit(articulation_vectors, images, epochs=epochs, shuffle=True)
        print("\n-- done seeding interpreter {}".format(self.name))

    def get_articulation_from_image(self, image):
        image = image.reshape((1, self.image_vector_len))
        articulation = self.eye.model.predict(image)
        return articulation

    def describe_image(self, image, add_noise=True):
        # should add noise when playing the game or talking to oneself, but not when showing the spectrum representation that has been learned
        articulation = self.get_articulation_from_image(image)
        spectrum_flattened = self.mouth.pronounce(articulation, add_noise=add_noise)
        spectrum_as_x = spectrum_flattened.reshape((1, self.full_spectrum_vector_len))
        # also use this as an ear-training sample
        self.ear.model.fit(spectrum_as_x, articulation)
        return spectrum_flattened

    def fit_spectrum_to_image(self, spectrum, image):
        # the image is the ultimate answer, backpropagate through the networks
        image = image.reshape((1, self.image_vector_len))
        spectrum = spectrum.reshape((1, self.full_spectrum_vector_len))
        articulation = self.ear.model.predict(spectrum)
        self.interpreter.model.fit(articulation, image)
 

class Eye:
    def __init__(self, eye_input_layer, eye_hidden_layers, eye_output_layer):
        self.input_layer = eye_input_layer
        self.hidden_layers = eye_hidden_layers
        self.output_layer = eye_output_layer
        self.model = keras.Model(eye_input_layer, eye_output_layer)
        opt = keras.optimizers.Adam(learning_rate=1e-5)
        self.model.compile(optimizer=opt, loss="mean_squared_error")


class Ear:
    def __init__(self, ear_input_layer, ear_hidden_layers, ear_output_layer):
        self.input_layer = ear_input_layer
        self.hidden_layers = ear_hidden_layers
        self.output_layer = ear_output_layer
        self.model = keras.Model(ear_input_layer, ear_output_layer)
        opt = keras.optimizers.Adam(learning_rate=1e-5)
        self.model.compile(optimizer=opt, loss="mean_squared_error")


class Interpreter:
    def __init__(self, ip_input_layer, ip_hidden_layers, ip_output_layer):
        self.input_layer = ip_input_layer
        self.hidden_layers = ip_hidden_layers
        self.output_layer = ip_output_layer
        self.model = keras.Model(ip_input_layer, ip_output_layer)
        opt = keras.optimizers.Adam(learning_rate=1e-5)
        self.model.compile(optimizer=opt, loss="binary_crossentropy")  # image output needs binary cross-entropy


class Mouth:
    def __init__(self, articulators, n_articulation_positions_per_sequence, noise_average_amplitude):
        self.frames_per_vector = 1
        self.n_articulation_positions_per_sequence = n_articulation_positions_per_sequence
        assert 0 <= noise_average_amplitude <= 1
        self.noise_average_amplitude = noise_average_amplitude

        self.articulators = articulators
        self.single_artv_len = self.get_single_articulation_vector_length()
        self.full_artv_len = self.get_full_articulation_vector_length()
        self.single_specv_len = self.get_single_spectrum_vector_length()
        self.full_specv_len = self.get_full_spectrum_vector_length()

    def get_random_single_articulation_vector(self):
        artv = bs.get_random_articulation_vectors(self.articulators, n_vectors=1)
        return np.array(artv)

    def get_random_full_articulation_vector(self, dimensions):
        # input("mouth has {} articulators".format(len(self.articulators)))
        artvs = bs.get_random_articulation_vectors(self.articulators, n_vectors=self.n_articulation_positions_per_sequence)
        artvs = np.array(artvs)
        # input("artvs shape {}".format(artvs.shape))
        if dimensions == 1:
            return artvs.reshape((artvs.size,))
        elif dimensions == 2:
            return artvs.reshape((1, artvs.size))
        else:
            raise ValueError("bad dimensions {}".format(dimensions))

    def get_single_articulation_vector_length(self):
        artv = self.get_random_single_articulation_vector()
        return artv.size

    def get_full_articulation_vector_length(self):
        artv = self.get_random_full_articulation_vector(dimensions=1)
        return artv.size

    def get_single_spectrum_vector_length(self):
        articulation_vector = self.get_random_single_articulation_vector()
        # full_articulation_vector = full_articulation_vector.reshape((full_articulation_vector.size,))
        spectrum_vector = self.pronounce(articulation_vector)
        return spectrum_vector.size

    def get_full_spectrum_vector_length(self):
        full_articulation_vector = self.get_random_full_articulation_vector(dimensions=1)
        spectrum_vector = self.pronounce(full_articulation_vector)
        return spectrum_vector.size

    def convert_full_spectrum_vector_to_spectrum_sequence(self, full_spectrum_vector):
        # the spectrum vector may actually represent multiple points in time, a sequence of articulations
        if len(full_spectrum_vector.shape) == 1:
            input_specv_len, = full_spectrum_vector.shape
        elif len(full_spectrum_vector.shape) == 2:
            one, input_specv_len = full_spectrum_vector.shape
            full_spectrum_vector = articulation_vector.reshape((input_specv_len,))  # get rid of the single-sample row dimension
            assert one == 1, "invalid specv shape: {}".format(full_spectrum_vector.shape)
        else:
            raise ValueError("invalid specv shape: {}".format(full_spectrum_vector.shape))
        assert input_specv_len % self.single_specv_len == 0, "spectrum vector wrong size, needed multiple of {}, got {}".format(self.single_specv_len, input_specv_len)
        n_sections = input_specv_len // self.single_specv_len
        spectrum_vectors = []
        for i in range(n_sections):
            section = full_spectrum_vector[self.single_specv_len*i : self.single_specv_len*(i+1)]
            spectrum_vectors.append(section)
        return np.array(spectrum_vectors)

    def convert_full_articulation_vector_to_articulation_sequence(self, full_articulation_vector):
        # the articulation vector may actually represent multiple points in time, a sequence of articulations
        if len(full_articulation_vector.shape) == 1:
            input_artv_len, = full_articulation_vector.shape
        elif len(full_articulation_vector.shape) == 2:
            one, input_artv_len = full_articulation_vector.shape
            assert one == 1, "invalid artv shape: {}".format(full_articulation_vector.shape)
            full_articulation_vector = full_articulation_vector.reshape((input_artv_len,))  # get rid of the single-sample row dimension
        else:
            raise ValueError("invalid artv shape: {}".format(full_articulation_vector.shape))
        assert input_artv_len % self.single_artv_len == 0, "articulation vector wrong size, needed multiple of {}, got {}".format(self.single_artv_len, input_artv_len)
        n_sections = input_artv_len // self.single_artv_len
        articulation_vectors = []
        for i in range(n_sections):
            section = full_articulation_vector[self.single_artv_len*i : self.single_artv_len*(i+1)]
            articulation_vectors.append(section)
        return np.array(articulation_vectors)


    def pronounce(self, articulation_vector, add_noise=True):
        # expects a flat vector like a neural net's output
        # should add noise when playing the game or talking to oneself, but not when showing the spectrum representation that has been learned
        articulation_vectors = self.convert_full_articulation_vector_to_articulation_sequence(articulation_vector)
        assert articulation_vectors.size == articulation_vector.size, "artv size was not conserved"
        # print(articulation_vectors)
        # input("L232")
        spectra = bs.get_spectra_from_vectors_in_articulation(articulation_vectors, self.articulators, frames_per_vector=self.frames_per_vector)
        # print(spectra)
        # input("L235")
        if add_noise:
            spectra_with_noise = bs.add_noise_to_spectra(spectra, noise_average_amplitude=self.noise_average_amplitude)
            assert spectra_with_noise.size == spectra.size, "spectra size was not conserved"
            spectra = spectra_with_noise
        spectra = bs.normalize_spectrum_vectors_to_01(spectra)
        spectrum_vector = np.array(spectra).reshape((spectra.size,))
        return spectrum_vector


def round_array_to_precision(arr, precision):
    return precision * np.round(arr / precision)


def round_array_to_n_ticks_01(arr, n_ticks):
    # one tick at 0, one at 1, rest evenly spaced between those
    assert (0 <= arr).all() and (arr <= 1).all(), "arr not in 01"
    assert type(n_ticks) is int
    assert n_ticks >= 2
    precision = 1/(n_ticks-1)  # fencepost
    return round_array_to_precision(arr, precision)


def get_receptor_category_ints(arr, n_receptors):
    # e.g. receptors at values [0, 0.2, 0.2, 0.6] with 6 receptors will give you [0, 1, 1, 3]
    return np.round(arr * (n_receptors-1)).astype(int)


def get_receptor_category_one_hot_stacked_from_01(arr, n_receptors):
    receptor_category_ints = get_receptor_category_ints(arr, n_receptors)
    return get_receptor_category_one_hot_stacked_from_indices(receptor_category_ints, n_receptors)


def get_receptor_category_one_hot_stacked_from_indices(arr, n_receptors):
    n_words, n_articulators = arr.shape
    unstacked = to_categorical(arr, num_classes=n_receptors)

    n_words2, n_articulators2, n_receptors2 = unstacked.shape
    assert n_words2 == n_words
    assert n_articulators2 == n_articulators
    assert n_receptors2 == n_receptors
    
    new_shape = (unstacked.shape[0], np.prod(unstacked.shape[1:]))
    stacked = unstacked.reshape(new_shape)
    n_words3, vec_len = stacked.shape
    assert n_words3 == n_words
    assert vec_len == n_articulators * n_receptors
    return stacked


def get_random_pronunciations(n_samples, n_articulation_positions, receptors_per_articulator):
    # return them as already one-hot encoded
    indices_arr = np.random.randint(0, n_articulation_positions, (n_samples, 5))
    one_hot = get_receptor_category_one_hot_stacked_from_indices(indices_arr, n_receptors=receptors_per_articulator)
    res = one_hot
    assert res.shape == (n_samples, 5 * n_articulation_positions)
    return res


def play_game(agents, images, n_rounds, images_per_turn):
    print("\n-- playing game with {} for {} rounds".format(agents, n_rounds))
    for round_i in range(n_rounds):
        print("playing game round {}".format(round_i))
        for agent_i in range(len(agents)):
            describer = agents[agent_i]
            # guessers = [agents[i] for i in range(len(agents)) if i != agent_i]
            for image_i in range(images_per_turn):
                image = random.choice(images)
                spectrum = describer.describe_image(image)
                for participant in agents:
                    # the describer should also train their other networks on what they said
                    # guess = guesser.guess_image_from_spectrum(spectrum)
                    participant.fit_spectrum_to_image(spectrum, image)
    print("\n-- done playing game")


def get_subsample(x, n_samples):
    total_n_samples = len(x)
    indices = random.sample(list(range(total_n_samples)), n_samples)
    samples = x[indices]
    return samples


def show_articulations_and_spectra_simple(articulations, spectra):
    n_articulations, single_artv_len = articulations.shape
    n_spectra, single_specv_len = spectra.shape
    assert n_articulations == n_spectra
    n_rows = n_spectra
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols)
    for row_i in range(n_spectra):
        art_ax = axes[row_i, 0]
        spec_ax = axes[row_i, 1]
        articulation = articulations[row_i, :]
        spectrum = spectra[row_i, :]
        title_this_iter = row_i == 0
        tick_labels_this_iter = row_i == n_spectra-1
        bs.plot_articulation(articulation, show=False, title=title_this_iter, tick_labels=tick_labels_this_iter, ax=art_ax)
        bs.plot_spectrum(spectrum, show=False, title=title_this_iter, tick_labels=tick_labels_this_iter, ax=spec_ax)
    plt.show()


def show_articulations_and_spectra_for_images(agents, images, n_images, show=True, save_sound=False, save_plot=False):
    print("\n-- saving/showing output articulations and spectra")
    images = get_subsample(images, n_images)
    for i, image in enumerate(images):
        show_articulations_and_spectra_for_image(agents, image, show=False, title=True, save_sound=save_sound, image_label=str(i), save_plot=save_plot)
        if show:
            plt.show()
    print("\n-- done saving/showing output articulations and spectra")


def show_articulations_and_spectra_for_image(agents, image, show=True, title=True, tick_labels=True, save_sound=False, image_label=None, save_plot=False):
    n_articulation_positions_per_sequence = agents[0].mouth.n_articulation_positions_per_sequence
    n_rows = 1 + n_articulation_positions_per_sequence  # one row for the image itself, one each for each segment in the word
    n_cols = 2  # left column for articulations, right column for spectra

    fig, axes = plt.subplots(n_rows, n_cols)

    # imshow the image array
    axes[0,0].imshow(image)
    axes[0,0].axis("off")  # just show image without xy ticks
    axes[0,1].axis("off")  # don't show this one, since there's nothing there

    # show the spectrum components
    for agent_i, agent in enumerate(agents):
        preset_colors = ["r", "b", "g", "k"]
        if agent_i < len(preset_colors):
            agent_color_rgba = preset_colors[agent_i]
        else:
            agent_color_rgba = tuple(np.random.uniform(0, 1, (3,))) + (0.75,)
        assert n_articulation_positions_per_sequence == agent.mouth.n_articulation_positions_per_sequence

        agent_raw_articulation_vector = agent.get_articulation_from_image(image)
        agent_articulation_sequence = agent.mouth.convert_full_articulation_vector_to_articulation_sequence(agent_raw_articulation_vector)
        n_segments = agent_articulation_sequence.shape[0]
        assert n_segments == agent.n_articulation_positions_per_sequence, "articulation sequence has {} segments but should have {}:\n{}".format(n_segments, agent.n_articulation_positions_per_sequence, agent_articulation_sequence)
        with open("BasisSpectrumOutput/BasisSpectrumArticulation_image{}_agent{}.txt".format(image_label, agent_i), "w") as f:
            for seg_i in range(n_segments):
                artv = agent_articulation_sequence[seg_i]
                vec_str = "[" + ", ".join("{:.4f}".format(x) for x in artv) + "]"
                f.write("segment {}: {}\n".format(seg_i, vec_str))

        agent_full_articulation_vector = agent_articulation_sequence.reshape((agent_articulation_sequence.size,))
        agent_full_spectrum_vector = agent.mouth.pronounce(agent_full_articulation_vector, add_noise=True)
        agent_spectrum_sequence = agent.mouth.convert_full_spectrum_vector_to_spectrum_sequence(agent_full_spectrum_vector)
        if save_sound:
            assert image_label is not None
            sound_fp = "BasisSpectrumOutput/BasisSpectrumOutput_image{}_agent{}.wav".format(image_label, agent_i)
            signal = bs.convert_spectrum_sequence_to_waveform(agent_spectrum_sequence, seconds=2)
            wav.write_signal_to_wav(signal, sound_fp)
        for articulation_i in range(n_articulation_positions_per_sequence):
            articulation_ax = axes[articulation_i+1, 0]
            spectrum_ax = axes[articulation_i+1, 1]
            title_this_iter = articulation_i == 0 and title
            tick_labels_this_iter = articulation_i == n_articulation_positions_per_sequence-1 and tick_labels
            articulation = agent_articulation_sequence[articulation_i]
            bs.plot_articulation(articulation, show=show, title=title_this_iter, tick_labels=tick_labels_this_iter, color=agent_color_rgba, ax=articulation_ax)
            spectrum = agent_spectrum_sequence[articulation_i]
            bs.plot_spectrum(spectrum, show=show, title=title_this_iter, tick_labels=tick_labels_this_iter, color=agent_color_rgba, ax=spectrum_ax)
    if save_plot:
        assert image_label is not None
        plt.savefig("BasisSpectrumOutput/BasisSpectrumPlot_image{}.png".format(image_label))
    if show:
        plt.show()
    plt.close()


def play_game_simple(initial_agents, new_agent, n_rounds_initial, n_rounds_with_new_learner, n_samples_per_round, epochs_per_round, learner_acceptance_threshold):
    agreement_proportions = []
    average_distances = []
    phases = ["initial", "new_learner"]
    for phase in phases:
        if phase == "initial":
            n_rounds = n_rounds_initial
            agents = initial_agents
        elif phase == "new_learner":
            n_rounds = n_rounds_with_new_learner
            agents = initial_agents + [new_agent]
        else:
            raise Exception(f"invalid phase {phase}")

        for round_i in range(n_rounds):
            print(f"\nround {round_i}/{n_rounds}")
            for agent_i, describer in enumerate(agents):
                print(f"current describer: {describer}")
                inputs = np.random.random((n_samples_per_round,))  # each input is a "card" containing a number from 0 to 1
                agent_production = describer.describe(inputs)

                if describer is new_agent:
                    if learner_acceptance_threshold is None:
                        older_agents_will_listen = True
                    else:
                        # they ignore the learner when it hasn't yet achieved some accuracy in learning the language
                        distance_items = get_agent_distances(agents)
                        distances_involving_learner = [d for a,b,d in distance_items if a is new_agent or b is new_agent]
                        learner_distance = np.mean(distances_involving_learner)
                        older_agents_will_listen = learner_distance < learner_acceptance_threshold
                        # can go back and forth, e.g. if the learner gets farther away again for some reason then they will not listen to it
                else:
                    older_agents_will_listen = True

                if older_agents_will_listen:
                    listeners = [a for a in agents if a is not describer]
                else:
                    print(f"older agents are ignoring the learner because {learner_distance} does not meet threshold {learner_acceptance_threshold}")
                    listeners = []

                for listener in listeners:
                    listener.perceive(agent_production, inputs, epochs=epochs_per_round)  # update perception and production models
            print("\nconventions this round:")
            agreement_proportion = report_form_meaning_correspondences(agents)
            agreement_proportions.append(agreement_proportion)
            average_distance = report_agent_distances(agents)
            average_distances.append(average_distance)

    plt.plot(agreement_proportions)
    plt.title("agreement proportion")
    plt.xlabel("round number")
    plt.savefig("/home/wesley/programming/BasisSpectrumOutput/agreement_proportion.png")
    plt.gcf().clear()

    plt.plot(average_distances)
    plt.title("average distance between agents")
    plt.xlabel("round number")
    plt.savefig("/home/wesley/programming/BasisSpectrumOutput/average_distance.png")
    plt.gcf().clear()


def report_form_meaning_correspondences(agents):
    arr = []
    agreements = []
    print("value " + " ".join(f"agt{i}" for i in range(len(agents))) + " same?")  # header
    for agent in agents:
        ms_ps = agent.get_pronunciations_of_meanings()
        meanings = [tup[0] for tup in ms_ps]
        pronunciations = [tup[1] for tup in ms_ps]
        arr.append(pronunciations)
    for meaning_i in range(len(arr[0])):
        s = f"{meanings[meaning_i]:5} "
        pronunciations = [arr[i][meaning_i] for i in range(len(agents))]
        pronunciations_all_same = len(set(pronunciations)) == 1
        agreements.append(pronunciations_all_same)
        pronunciations_str = " ".join(pronunciations)
        all_same_str = "same!" if pronunciations_all_same else "***"
        s += f"{pronunciations_str} {all_same_str}"
        print(s)
    agreement_proportion = np.mean(agreements)
    return agreement_proportion


def get_agent_distances(agents):
    language_arrays = {a: a.get_language_vector() for a in agents}
    combos = itertools.combinations(agents, 2)
    distances = []
    for a, b in combos:
        arr_a = language_arrays[a]
        arr_b = language_arrays[b]
        dist = np.linalg.norm(arr_a - arr_b)
        item = [a, b, dist]
        distances.append(item)
    return distances


def report_agent_distances(agents):
    distance_items = get_agent_distances(agents)
    distances = []
    for a, b, dist in distance_items:
        print(f"distance from {a} to {b} is {dist}")
        distances.append(dist)
    average_distance = np.mean(distances)
    return average_distance


def get_predetermined_categorization_seeding_data(n_samples, condition, n_articulation_positions, receptors_per_articulator):
    if condition == "green-blue":
        # 0 - 0.25 is A
        # 0.25 - 0.5 is B
        # 0.5 - 0.75 is C
        # 0.75 - 1 is D
        pronunciations_unassigned = get_random_pronunciations(4, n_articulation_positions, receptors_per_articulator)
        meanings = np.random.random((n_samples, 1))
        pronunciations = []
        for meaning in meanings:
            if 0 <= meaning < 0.25:
                pi = 0
            elif 0.25 <= meaning < 0.5:
                pi = 1
            elif 0.5 <= meaning < 0.75:
                pi = 2
            else:
                pi = 3
            pronunciations.append(pronunciations_unassigned[pi])
        pronunciations = np.array(pronunciations)

    elif condition == "grue":
        # 0 - 0.25 is E
        # 0.25 - 0.75 is F
        # 0.75 - 1 is G
        pronunciations_unassigned = get_random_pronunciations(3, n_articulation_positions, receptors_per_articulator)
        meanings = np.random.random((n_samples, 1))
        pronunciations = []
        for meaning in meanings:
            if 0 <= meaning < 0.25:
                pi = 0
            elif 0.25 <= meaning < 0.75:
                pi = 1
            else:
                pi = 2
            pronunciations.append(pronunciations_unassigned[pi])
        pronunciations = np.array(pronunciations)

    elif condition == "random":
        # different term for each of the seed meanings
        pronunciations = get_random_pronunciations(n_samples, n_articulation_positions, receptors_per_articulator)
        meanings = np.random.random((n_samples, 1))
    else:
        raise ValueError(f"invalid condition {condition}")
    return pronunciations, meanings


if __name__ == "__main__":
    # should call get_articulators() for each instance of Agent, so it's not pointing to the same objects among different agents (you can't have the same tongue as someone else)
    # mnist_vector_len = 28**2
    # (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
    n_initial_agents = 2
    n_articulation_positions = 26
    # n_articulation_positions_per_sequence = 3
    # noise_average_amplitude = 0
    # n_babble_samples = 100000
    # n_babble_epochs = 100
    # babble_batch_size = 100
    # n_eye_seed_samples = 1
    # n_eye_seed_epochs = 1
    # n_interpreter_seed_samples = 1
    # n_interpreter_seed_epochs = 1
    class_simulating_for = "langcog"
    n_rounds_initial = 250
    n_rounds_with_new_learner = 0 if class_simulating_for == "langcog" else 250
    n_seeding_samples = 250
    n_seeding_epochs = 1000
    # images_per_turn = 10
    # n_images_to_save = 100
    n_samples_per_round = 100
    epochs_per_round = 250
    learner_acceptance_threshold = 0.10
    agent_noise_stdev = 0 # 1/((n_articulation_positions-1)*3)

    initial_agents = []
    for i in range(n_initial_agents):
        print("creating agent #{}".format(i))
        name = "Agent{}".format(i)
        # arts = bs.get_articulators()
        # input("got {} arts".format(len(arts)))
        # a = Agent(name, arts, mnist_vector_len, n_articulation_positions_per_sequence, noise_average_amplitude)
        a = SimpleAgent.random(name, n_articulation_positions, noise_stdev=agent_noise_stdev)
        
        # babbling is true feedback, the real auditory spectrum made by articulation
        # a.babble(n_samples=n_babble_samples, epochs=n_babble_epochs, batch_size=babble_batch_size)

        # the eye and interpreter seeding is false feedback
        # just intended to get the model started on something non-degenerate that will later be overwritten by convention created among the agents
        # a.seed_eye(get_subsample(mnist_x_train, n_eye_seed_samples), epochs=n_eye_seed_epochs)
        # a.seed_interpreter(get_subsample(mnist_x_train, n_interpreter_seed_samples), epochs=n_interpreter_seed_epochs)

        if class_simulating_for == "theophon":
            seeding_condition = "random"  # "random" for theory of phonology
        elif class_simulating_for == "langcog":
            seeding_condition = "green-blue" if i % 2 == 0 else "grue"
        else:
            raise ValueError(f"unknown course {class_simulating_for}")

        a.seed(n_samples=n_seeding_samples, epochs=n_seeding_epochs, condition=seeding_condition)  # start them with some association so they don't just sit at the middle of the space the whole time
        initial_agents.append(a)

    new_agent = SimpleAgent.random("NewLearner", n_articulation_positions, noise_stdev=agent_noise_stdev)
    # DON'T seed the new agent, they will learn from the others

    # play_game(agents, mnist_x_train, n_rounds=n_rounds, images_per_turn=images_per_turn)
    print("agents' starting state:")
    report_form_meaning_correspondences(initial_agents)
    play_game_simple(initial_agents, new_agent, n_rounds_initial, n_rounds_with_new_learner, n_samples_per_round, epochs_per_round, learner_acceptance_threshold)
    # show_articulations_and_spectra_for_images(agents, mnist_x_train, n_images=n_images_to_save, save_sound=True, save_plot=True, show=False)
