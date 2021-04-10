# implementation of the neural net agents who learn to communicate with the articulations/spectra in BasisSpectra.py

import keras
from keras import layers
from keras.datasets import mnist
import BasisSpectra as bs
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import Music.WavUtil as wav


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


class Agent:
    def __init__(self, articulators, image_vector_len):
        self.articulators = articulators
        self.mouth = Mouth(articulators, n_articulation_positions_per_sequence=5, noise_average_amplitude=0.1)
        self.image_vector_len = image_vector_len
        self.single_articulation_vector_len = self.mouth.single_artv_len
        self.full_articulation_vector_len = self.mouth.full_artv_len
        self.single_spectrum_vector_len = self.mouth.single_specv_len
        self.full_spectrum_vector_len = self.mouth.full_specv_len

        eye_input_layer = keras.Input(shape=(self.image_vector_len,))
        eye_hl0 = layers.Dense(self.image_vector_len, activation="relu")(eye_input_layer)
        eye_hl1 = layers.Dense(self.image_vector_len, activation="relu")(eye_hl0)
        eye_hidden_layers = [eye_hl0, eye_hl1]
        # if the articulators expect articulator param values in [0, 1], then anything outputting articulation vector should have activation of sigmoid
        eye_output_layer = layers.Dense(self.full_articulation_vector_len, activation="sigmoid")(eye_hidden_layers[-1])
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

    def babble(self, samples, epochs, batch_size):
        x_train, y_train = self.create_babble_dataset_for_ear(samples)
        x_test, y_test = self.create_babble_dataset_for_ear(max(10, int(samples*0.1)))
        print("x_train shape {}, y_train shape {}".format(x_train.shape, y_train.shape))
        self.ear.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test))

    def create_babble_dataset_for_ear(self, batch_size):
        x = []
        y = []
        for i in range(batch_size):
            this_x, this_y = self.create_single_babble_data_point_for_ear()
            x.append(this_x)
            y.append(this_y)
        return np.array(x), np.array(y)

    def create_single_babble_data_point_for_ear(self):
        articulation_vector = self.mouth.get_random_full_articulation_vector()
        spectrum = self.mouth.pronounce(articulation_vector)
        return (spectrum, articulation_vector)

    def get_random_seed_articulation_vectors_for_images(self, images):
        n_images, *single_image_shape = images.shape
        image_vector_len = np.prod(single_image_shape)
        assert image_vector_len == self.image_vector_len
        articulation_vectors = [self.mouth.get_random_full_articulation_vector(dimensions=1) for i in range(n_images)]
        articulation_vectors = np.array(articulation_vectors)
        return articulation_vectors

    def seed_eye(self, images):
        # give the Eye just a few images along with random articulations to train on, so it's not starting from nothing
        articulation_vectors = self.get_random_seed_articulation_vectors_for_images(images)
        images = images.reshape(images.shape[0], self.image_vector_len)
        self.eye.model.fit(images, articulation_vectors, epochs=20, shuffle=True)

    def seed_interpreter(self, images):
        # give the interpreter a few random articulation vectors and images to train on, so it's not starting from nothing
        articulation_vectors = self.get_random_seed_articulation_vectors_for_images(images)
        images = images.reshape(images.shape[0], self.image_vector_len)
        self.interpreter.model.fit(articulation_vectors, images, epochs=20, shuffle=True)

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
        self.model.compile(optimizer="adam", loss="mean_squared_error")


class Ear:
    def __init__(self, ear_input_layer, ear_hidden_layers, ear_output_layer):
        self.input_layer = ear_input_layer
        self.hidden_layers = ear_hidden_layers
        self.output_layer = ear_output_layer
        self.model = keras.Model(ear_input_layer, ear_output_layer)
        self.model.compile(optimizer="adam", loss="mean_squared_error")


class Mouth:
    def __init__(self, articulators, n_articulation_positions_per_sequence, noise_average_amplitude):
        self.frames_per_vector = 1
        self.n_articulation_positions_per_sequence = n_articulation_positions_per_sequence
        assert 0 <= noise_average_amplitude <= 1
        self.noise_average_amplitude = noise_average_amplitude

        self.articulators = articulators
        self.single_artv_len = self.get_single_articulation_vector_length()
        self.full_artv_len = self.n_articulation_positions_per_sequence * self.single_artv_len
        self.single_specv_len = self.get_single_spectrum_vector_length()
        self.full_specv_len = self.n_articulation_positions_per_sequence * self.single_specv_len

    def get_random_single_articulation_vector(self):
        artv, = bs.get_random_articulation_vectors(self.articulators, n_vectors=1)
        return np.array(artv)

    def get_random_full_articulation_vector(self, dimensions=2):
        artvs = bs.get_random_articulation_vectors(self.articulators, n_vectors=self.n_articulation_positions_per_sequence)
        artvs = np.array(artvs)
        if dimensions == 1:
            return artvs.reshape((artvs.size,))
        elif dimensions == 2:
            return artvs.reshape((1, artvs.size))
        else:
            raise ValueError("bad dimensions {}".format(dimensions))

    def get_single_articulation_vector_length(self):
        artv = self.get_random_single_articulation_vector()
        return len(artv)

    def get_single_spectrum_vector_length(self):
        articulation_vector = self.get_random_single_articulation_vector()
        spectrum_vector = self.pronounce(articulation_vector)
        return len(spectrum_vector)

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
            full_articulation_vector = full_articulation_vector.reshape((input_artv_len,))  # get rid of the single-sample row dimension
            assert one == 1, "invalid artv shape: {}".format(full_articulation_vector.shape)
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
        # should add noise when playing the game or talking to oneself, but not when showing the spectrum representation that has been learned
        articulation_vectors = self.convert_full_articulation_vector_to_articulation_sequence(articulation_vector)
        spectra = bs.get_spectra_from_vectors_in_articulation(articulation_vectors, self.articulators, frames_per_vector=self.frames_per_vector)
        if add_noise:
            spectra = bs.add_noise_to_spectra(spectra, noise_average_amplitude=self.noise_average_amplitude)
        spectrum_vector = np.array(spectra).flatten()
        return spectrum_vector


class Interpreter:
    def __init__(self, ip_input_layer, ip_hidden_layers, ip_output_layer):
        self.input_layer = ip_input_layer
        self.hidden_layers = ip_hidden_layers
        self.output_layer = ip_output_layer
        self.model = keras.Model(ip_input_layer, ip_output_layer)
        self.model.compile(optimizer="adam", loss="mean_squared_error")


def play_game(agents, images, n_rounds):
    image_i = 0
    for round_i in range(n_rounds):
        print("playing game round {}".format(round_i))
        for agent_i in range(len(agents)):
            describer = agents[agent_i]
            # guessers = [agents[i] for i in range(len(agents)) if i != agent_i]
            image = images[image_i]
            spectrum = describer.describe_image(image)
            for participant in agents:
                # the describer should also train their other networks on what they said
                # guess = guesser.guess_image_from_spectrum(spectrum)
                participant.fit_spectrum_to_image(spectrum, image)
            image_i += 1


def get_subsample(x, n_samples):
    total_n_samples = len(x)
    indices = random.sample(list(range(total_n_samples)), n_samples)
    samples = x[indices]
    return samples


def show_spectra_for_images(agents, images, n_images, show=True, save_sound=False, save_plot=False):
    images = get_subsample(images, n_images)
    for i, image in enumerate(images):
        show_spectra_for_image(agents, image, show=False, title=True, save_sound=save_sound, image_label=str(i), save_plot=save_plot)
        if show:
            plt.show()


def show_spectra_for_image(agents, image, show=True, title=True, save_sound=False, image_label=None, save_plot=False):
    n_articulation_positions_per_sequence = agents[0].mouth.n_articulation_positions_per_sequence
    n_rows = 1 + n_articulation_positions_per_sequence  # one row for the image itself, one each for each segment in the word
    n_cols = 1

    fig, axes = plt.subplots(n_rows, n_cols)

    # imshow the image array
    axes[0].imshow(image)

    # show the spectrum components
    for agent_i, agent in enumerate(agents):
        agent_color_rgba = tuple(np.random.uniform(0, 1, (3,))) + (0.75,)
        assert n_articulation_positions_per_sequence == agent.mouth.n_articulation_positions_per_sequence

        agent_full_articulation_vector = agent.get_articulation_from_image(image)
        agent_articulation_sequence = agent.mouth.convert_full_articulation_vector_to_articulation_sequence(agent_full_articulation_vector)
        with open("BasisSpectrumOutput/BasisSpectrumArticulation_image{}_agent{}.txt".format(image_label, agent_i), "w") as f:
            n_segments = agent_articulation_sequence.shape[0]
            for seg_i in range(n_segments):
                f.write("segment {}: {}\n".format(seg_i, agent_articulation_sequence[seg_i]))

        agent_full_spectrum_vector = agent.describe_image(image, add_noise=False)
        agent_spectrum_sequence = agent.mouth.convert_full_spectrum_vector_to_spectrum_sequence(agent_full_spectrum_vector)
        if save_sound:
            assert image_label is not None
            sound_fp = "BasisSpectrumOutput/BasisSpectrumOutput_image{}_agent{}.wav".format(image_label, agent_i)
            signal = bs.convert_spectrum_sequence_to_waveform(agent_spectrum_sequence, seconds=2)
            wav.write_signal_to_wav(signal, sound_fp)
        for articulation_i in range(n_articulation_positions_per_sequence):
            ax = axes[articulation_i+1]
            title = articulation_i == 0 and title
            spectrum = agent_spectrum_sequence[articulation_i]
            bs.plot_spectrum(spectrum, show=show, title=title, color=agent_color_rgba, ax=ax)
    if save_plot:
        assert image_label is not None
        plt.savefig("BasisSpectrumOutput/BasisSpectrumPlot_image{}.png".format(image_label))
    if show:
        plt.show()


if __name__ == "__main__":
    # should call get_articulators() for each instance of Agent, so it's not pointing to the same objects among different agents (you can't have the same tongue as someone else)
    mnist_vector_len = 28**2
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
    n_agents = 2
    agents = []
    for i in range(n_agents):
        print("creating agent #{}".format(i))
        arts = bs.get_articulators()
        a = Agent(arts, mnist_vector_len)
        a.babble(samples=2000, epochs=25, batch_size=100)  # babbling is true feedback, the real auditory spectrum made by articulation
        a.seed_eye(get_subsample(mnist_x_train, 10))
        # the eye and interpreter seeding is false feedback
        # just intended to get the model started on something non-degenerate that will later be overwritten by convention created among the agents
        a.seed_interpreter(get_subsample(mnist_x_train, 10))
        agents.append(a)

    play_game(agents, mnist_x_train, n_rounds=5)
    show_spectra_for_images(agents, mnist_x_train, n_images=2, save_sound=True, save_plot=True, show=False)
