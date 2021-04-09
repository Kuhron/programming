# implementation of the neural net agents who learn to communicate with the articulations/spectra in BasisSpectra.py

import keras
from keras import layers
from keras.datasets import mnist
import BasisSpectra as bs
import numpy as np
import matplotlib.pyplot as plt
import math
import random


# agent babbles in order to learn what its articulations sound like, so it can train its ear
# the Mouth is a device endowed upon the agent, cannot learn anything, implemented in BasisSpectra.py with Articulator classes
# don't bother with sound waves, just give them spectra directly, can add noise to them but don't bother with fft and ifft
# the Ear is a neural net from spectrum to articulation vector
# let articulation vector be the internal representation
# the Eye is a neural net from image to internal representation (articulation vector)
# and the Interpreter is a neural net from internal representation to image

# babbling task: agent creates random articulation vector, resulting spectrum (sound, with noise) is fed to the Ear, the Ear's predicted articulation vector is evaluated relative to the actual one
# multi-agent picture description task: agents take turns, in each turn, one agent (the Describer) sees image, the Eye creates internal representation from it, the Mouth says it, the resulting spectrum (with noise) is fed to all agents' (including the Describer's) Ears, they create articulation vectors using the Ear's prediction, then the Interpreter (again for all agents including the Describer) creates an image from the articulation vector which the Ear output, this image is then evaluated versus the actual one, train the Interpreter and Ear this way
# how is the Eye trained? all listeners (including the Describer itself) should take the articulatory representation spoken for that picture, use that and the actual image to train the Eye


class Agent:
    def __init__(self, articulators, image_vector_len):
        self.articulators = articulators
        self.mouth = Mouth(articulators)
        articulation_vector_len = self.mouth.artv_len
        spectrum_vector_len = self.mouth.specv_len

        eye_input_layer = keras.Input(shape=(image_vector_len,))
        eye_hidden_layers = [
            layers.Dense(image_vector_len, activation="relu")(eye_input_layer),
        ]
        eye_output_layer = layers.Dense(articulation_vector_len, activation="sigmoid")(eye_hidden_layers[-1])
        self.eye = Eye(eye_input_layer, eye_hidden_layers, eye_output_layer)

        ear_input_layer = keras.Input(shape=(spectrum_vector_len,))
        ear_hidden_layers = [
            layers.Dense(spectrum_vector_len, activation="relu")(ear_input_layer),
        ]
        ear_output_layer = layers.Dense(articulation_vector_len, activation="sigmoid")(ear_hidden_layers[-1])
        self.ear = Ear(ear_input_layer, ear_hidden_layers, ear_output_layer)

        self.interpreter = Interpreter()

    def babble(self, samples, epochs, batch_size):
        x_train, y_train = self.create_babble_dataset_for_ear(samples)
        x_test, y_test = self.create_babble_dataset_for_ear(max(10, int(samples*0.1)))
        print("x_train shape {}, y_train shape {}".format(x_train.shape, y_train.shape))
        self.ear.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    def create_babble_dataset_for_ear(self, batch_size):
        x_train = []
        y_train = []
        for i in range(batch_size):
            x, y = self.create_single_babble_data_point_for_ear()
            x_train.append(x)
            y_train.append(y)
        return np.array(x_train), np.array(y_train)

    def create_single_babble_data_point_for_ear(self):
        articulation_vector = self.mouth.get_random_articulation_vector()
        spectrum = self.mouth.pronounce(articulation_vector)
        return (spectrum, articulation_vector)
 

class Eye:
    def __init__(self, eye_input_layer, eye_hidden_layers, eye_output_layer):
        self.input_layer = eye_input_layer
        self.hidden_layers = eye_hidden_layers
        self.output_layer = eye_output_layer
        self.model = keras.Model(eye_input_layer, eye_output_layer)
        self.model.compile(optimizer="adam", loss="mean_squared_error")


class Ear:
    def __init__(self, ear_input_layer, ear_hidden_layers, ear_output_layer):
        self.input_layer = ear_input_layer
        self.hidden_layers = ear_hidden_layers
        self.output_layer = ear_output_layer
        self.model = keras.Model(ear_input_layer, ear_output_layer)
        self.model.compile(optimizer="adam", loss="mean_squared_error")


class Mouth:
    def __init__(self, articulators):
        self.frames_per_vector = 1
        self.articulators = articulators
        self.artv_len = self.get_articulation_vector_length()
        self.specv_len = self.get_spectrum_vector_length()

    def get_random_articulation_vector(self):
        artv, = bs.get_random_articulation_vectors(self.articulators, n_vectors=1)
        return artv

    def get_articulation_vector_length(self):
        artv = self.get_random_articulation_vector()
        return len(artv)

    def get_spectrum_vector_length(self):
        articulation_vector = self.get_random_articulation_vector()
        spectrum_vector = self.pronounce(articulation_vector)
        return len(spectrum_vector)

    def pronounce(self, articulation_vector):
        # the articulation vector may actually represent multiple points in time, a sequence of articulations
        self_artv_len = self.artv_len
        input_artv_len, = articulation_vector.shape
        assert input_artv_len % self_artv_len == 0, "articulation vector wrong size, needed multiple of {}, got {}".format(self_artv_len, input_artv_len)
        n_sections = input_artv_len // self_artv_len
        articulation_vectors = []
        for i in range(n_sections):
            section = articulation_vector[self_artv_len*i : self_artv_len*(i+1)]
            articulation_vectors.append(section)
        spectra = bs.get_spectra_from_vectors_in_articulation(articulation_vectors, self.articulators, frames_per_vector=self.frames_per_vector)
        spectra = bs.add_noise_to_spectra(spectra, noise_average_amplitude=0.5)
        spectrum_vector = np.array(spectra).flatten()
        return spectrum_vector


class Interpreter:
    def __init__(self):
        pass



if __name__ == "__main__":
    # should call get_articulators() for each instance of Agent, so it's not pointing to the same objects among different agents (you can't have the same tongue as someone else)
    n_agents = 1
    agents = []
    mnist_vector_len = 28**2
    for i in range(n_agents):
        arts = bs.get_articulators()
        a = Agent(arts, mnist_vector_len)
        agents.append(a)
    agents[0].babble(samples=1000, epochs=100, batch_size=200)
