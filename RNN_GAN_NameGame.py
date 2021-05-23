# try making GAN architecture with RNNs on the names that I've generated with NameMaker Flask app

from silence_tensorflow import silence_tensorflow
silence_tensorflow()  # these warnings are super annoying and useless
import tensorflow as tf

import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random



def get_names(names_fps):
    names = []
    for names_fp in names_fps:
        with open(names_fp) as f:
            lines = f.readlines()
        this_file_names = [line.split(" : ")[1] for line in lines]
        names += this_file_names
    return names


def get_chars_from_names(names):
    chars = set()
    for name in names:
        chars |= set(name)
    return sorted(chars)


def pad_name(name, desired_len, pad_char):
    # pad with trailing chars that aren't in the input
    assert pad_char not in name
    return name.ljust(desired_len, pad_char)


def encode_name(name, chars):
    res = []
    for c in name:
        char_vec = encode_char(c, chars)
        res.append(char_vec)
    return np.array(res)


def encode_char(c, chars):
    res = [int(c2 == c) for c2 in chars]
    if c not in chars:
        assert all(x == 0 for x in res)  # for padding char, want to give arr of all zero so Masking layer can recognize it as invalid
    return res


def get_real_names(n, names, chars):
    res = []
    res_strs = []
    for i in range(n):
        name = random.choice(names)  # allow repetition
        encoding = encode_name(name, chars)
        res.append(encoding)
        res_strs.append(name)
        # print(f"got real name: {name} (len {len(name)})")
    return np.array(res), res_strs


def get_random_fake_names(n, name_len, chars):
    res = []
    res_strs = []
    for name_i in range(n):
        name_arr, name_str = get_single_random_fake_name(name_len, chars)
        res.append(name_arr)
        res_strs.append(name_str)
    return np.array(res), res_strs


def get_single_random_fake_name(max_name_len, chars):
    # using random char selection, NOT using the generator network
    res = []
    res_str = ""
    this_name_len = random.randint(max_name_len//4, max_name_len)
    for i in range(this_name_len):
        char_i = random.randrange(len(chars))
        char_vec = [int(j == char_i) for j in range(len(chars))]
        res.append(char_vec)
        res_str += chars[char_i]
    # print("got random fake name: {} (len {})".format(res_str, len(res_str)))
    assert len(res_str) == this_name_len
    assert len(res) == this_name_len
    return np.array(res), res_str


def generate_latent_points(generator_model, n_samples, chars):
    # using the generator model
    input_shape = generator_model.layers[0].input_shape
    # print(f"input_shape = {input_shape}")
    assert input_shape[0] is None, input_shape
    _, timesteps, input_vec_len = input_shape
    input_array_shape_to_create = (n_samples, timesteps, input_vec_len)
    vec = np.random.normal(0, 1, input_array_shape_to_create)
    return vec


def generate_fake_samples_from_latent_points(generator_model, n_samples, chars):
    vec = generate_latent_points(generator_model, n_samples, chars)
    output = generator_model.predict(vec)
    
    fake_input_for_discriminator = output
    fake_class_labels = np.array([0 for i in range(n_samples)])  # the class labels are not themselves fake, they are honest in saying that this data is fake, so the discriminator will be able to learn
    return fake_input_for_discriminator, fake_class_labels


def generate_fake_names_from_latent_points(generator_model, n_samples, chars, padding_char):
    vec = generate_latent_points(generator_model, n_samples, chars)
    output = generator_model.predict(vec)

    res = []
    for output_sample in output:
        name = convert_char_vector_sequence_to_name(output_sample, chars, padding_char)
        res.append(name)
    return res


def convert_char_vector_sequence_to_name(char_vecs, chars, padding_char):
    res = ""
    for char_vec in char_vecs:
        c = convert_char_vector_to_char(char_vec, chars, padding_char)
        res += c
        # print(f"char_vec {char_vec} yielded char {c}")
    return res


def convert_char_vector_to_char(char_vec, chars, padding_char):
    if (char_vec == 0).all():
        return padding_char
    else:
        max_val = char_vec.max()
        options = np.array(chars)[char_vec == max_val]  # the str chars at indices where the float vector has max value
        return np.random.choice(options)


def generate_random_fake_samples(n_samples, name_len, chars):
    name_arr, name_strs = get_random_fake_names(n_samples, name_len, chars)
    y = np.array([0 for x in range(n_samples)])
    return name_arr, y


def generate_real_samples(n_samples, names, chars):
    name_arr, name_strs = get_real_names(n_samples, names, chars)
    y = np.array([1 for x in range(n_samples)])  # class labels that these are real, for the discriminator to learn
    return name_arr, y


def shuffle_iterables_same_order(iterables):
    length = len(iterables[0])
    assert all(len(x) == length for x in iterables), "incompatible lengths"
    sample_indices = random.sample(list(range(length)), length)

    new_iterables = []
    for iterable in iterables:
        if type(iterable) is np.ndarray:
            new_iterable = iterable[sample_indices]
        elif type(iterable) is list:
            new_iterable = [iterable[i] for i in sample_indices]
        else:
            raise TypeError("unknown iterable type to shuffle: {}".format(type(iterable)))

        new_iterables.append(new_iterable)
    return new_iterables


def train_discriminator_initial(discriminator_model, names, chars):
    name_len = len(names[0])
    assert all(len(name) == name_len for name in names), "names have not been padded"
    n_samples = 1000
    n_batches = 15
    for batch_i in range(n_batches):
        print(f"discriminator training batch {batch_i}/{n_batches}")
        real_names, real_name_strs = get_real_names(n_samples, names, chars)
        fake_names, fake_name_strs = get_random_fake_names(n_samples, name_len, chars)  # start discriminator off learning from random, not from generator, but later should train it on generator's output
        batch_names = np.concatenate([real_names, fake_names])
        assert batch_names.shape == (2*n_samples, name_len, len(chars)), batch_names.shape
        batch_name_strs = real_name_strs + fake_name_strs  # just for display purposes, not for the model
        batch_outputs = np.array([1 for x in real_names] + [0 for x in fake_names])

        batch_names, batch_outputs, batch_name_strs = shuffle_iterables_same_order([batch_names, batch_outputs, batch_name_strs])
        # shuffle sample inputs and outputs into same order

        print("name strs this batch:")
        for x in batch_name_strs:
            print(repr(x))

        discriminator_model.fit(batch_names, batch_outputs)


def create_combined_gan(generator_model, discriminator_model):
    # from tutorial at https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
    discriminator_model.trainable=False
    model = keras.Sequential(name="gan_model")
    model.add(generator_model)
    model.add(discriminator_model)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_gan(generator_model, discriminator_model, gan_model, names, chars, n_epochs, batch_size, padding_char, plot_loss=False):
    n_samples = batch_size  # for the various generation functions
    n_batches = len(names) // batch_size

    if plot_loss:
        discriminator_loss_real_history = []
        discriminator_loss_fake_history = []
        generator_loss_history = []
        plt.ion()
        window_length = 1000  # for plotting losses

    for epoch_i in range(n_epochs):
        print(f"GAN epoch {epoch_i}/{n_epochs}")
        for batch_i in range(n_batches):
            X_real, Y_real = generate_real_samples(n_samples, names, chars)
            # print(f"XYreal {X_real.shape}, {Y_real.shape}")
            name_timesteps = X_real.shape[1]
            X_fake, Y_fake = generate_fake_samples_from_latent_points(generator_model, n_samples, chars)
            # print(f"XYfake {X_fake.shape}, {Y_fake.shape}")
            # X = np.concatenate([X_real, X_fake])
            # Y = np.concatenate([Y_real, Y_fake])
            # X, Y = shuffle_iterables_same_order([X, Y])

            # try training discriminator on real and fake separately; somehow it seems that combining them into a batch can cause failure to converge: https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
            discriminator_loss_real = discriminator_model.train_on_batch(X_real, Y_real)
            discriminator_loss_fake = discriminator_model.train_on_batch(X_fake, Y_fake)

            X_gan = generate_latent_points(generator_model, n_samples, chars)
            # print(f"X_gan {X_gan.shape}")
            Y_gan = np.array([1 for i in range(n_samples)])
            # tutorial says: "update the generator via the discriminator's error"  # oh, I see. You want the generator to take random noise inputs and try to create outputs of 1 in the discriminator (i.e., fool it)
            generator_loss = gan_model.train_on_batch(X_gan, Y_gan)

            print(f"discriminator loss real = {discriminator_loss_real:.4f}; fake = {discriminator_loss_fake:.4f}; generator loss = {generator_loss:.4f}", end="\r")

            # every batch:
            if plot_loss:
                discriminator_loss_real_history.append(discriminator_loss_real)
                discriminator_loss_fake_history.append(discriminator_loss_fake)
                generator_loss_history.append(generator_loss)

        # every epoch:
        print()  # so the next line doesn't overwrite the last loss line which ends with \r
        if True: # discriminator_loss_fake > 0.5 or epoch_i % 5 == 0:
            show_novel_names(generator_model, n_novel_names=10, padding_char=padding_char)

        if plot_loss:
            discriminator_loss_real_history = discriminator_loss_real_history[-window_length:]
            discriminator_loss_fake_history = discriminator_loss_fake_history[-window_length:]
            generator_loss_history = generator_loss_history[-window_length:]
    
            plt.gcf().clear()
            plt.plot(discriminator_loss_real_history, c="b")
            plt.plot(discriminator_loss_fake_history, c="r")
            plt.plot(generator_loss_history, c="y")
            plt.draw()
            plt.pause(0.01)

    plt.ioff()


def show_novel_names(generator_model, n_novel_names, padding_char):
    novel_names = generate_fake_names_from_latent_points(generator_model, n_novel_names, chars, padding_char)
    for x in novel_names:
        print(f"novel name generated: {x}")       



if __name__ == "__main__":
    names_fps = [
        # "/home/wesley/Desktop/Construction/NameMakerRatings.txt",
        "/home/wesley/Desktop/Construction/NameMakerRatings-2.txt",
    ]
    names = get_names(names_fps)
    chars = get_chars_from_names(names)
    n_chars = len(chars)
    padding_char = "#"
    assert " " in chars  # we DO want space to be considered part of a name
    assert padding_char not in chars

    padded_name_len = max(len(name) for name in names)
    n_timesteps = padded_name_len
    names = [pad_name(name, padded_name_len, padding_char) for name in names]

    mask = False  # should discriminator ignore spaces?

    # generator
    generator_input_vector_len = 100  # idk, something random
    generator_input_shape = (n_timesteps, generator_input_vector_len)
    generator_output_vector_len = len(chars)

    generator_input_layer = layers.Input(generator_input_shape, name="generator_input")
    generator_recurrent = layers.LSTM(64, activation="relu", name="generator_rnn", return_sequences=True)
    generator_output_layer = layers.Dense(generator_output_vector_len, activation="sigmoid", name="generator_output")

    generator_model = keras.Sequential(name="generator")
    generator_model.add(generator_input_layer)  # add one-by-one for debugging purposes
    generator_model.add(generator_recurrent)
    generator_model.add(generator_output_layer)

    # NOTE: generator is never trained by itself! it's always based on the discriminator output in combined GAN
    # generator_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    # generator_model.compile(optimizer=generator_optimizer, loss="mean_squared_error")

    # discriminator
    discriminator_input_vector_len = n_chars
    discriminator_input_shape = (n_timesteps, discriminator_input_vector_len)
    discriminator_output_vector_len = 1  # real/fake

    discriminator_input_layer = layers.Input(shape=discriminator_input_shape, name="discriminator_input")
    batch_shape = (None, n_timesteps, discriminator_input_vector_len)
    discriminator_recurrent = layers.LSTM(64, activation="relu", name="discriminator_rnn", return_sequences=False)
    # don't return sequences from the discriminator's recurrent layer because the loss function needs to compare the true value (0 or 1) with a SINGLE value from the RNN, NOT a sequence of values; the "can not squeeze" error is due to this mismatch where the loss is trying to compare a sequence of outputs to a single-timestep output
    discriminator_output_layer = layers.Dense(discriminator_output_vector_len, activation="sigmoid", name="discriminator_output")

    discriminator_model = keras.Sequential(name="discriminator")
    discriminator_model.add(discriminator_input_layer)  # add one-by-one for debugging purposes

    if mask:
        masking = layers.Masking(mask_value=0.0, batch_input_shape=batch_shape)
        # Masking should block it from paying attention to trailing chars after variable length input; if all cells in the input at a certain time step are equal to the mask value, then it will be ignored for that timestep (so make them all 0, in contrast to normal timesteps which have one-hot character encoding)
        discriminator_model.add(masking)

    discriminator_model.add(discriminator_recurrent)
    discriminator_model.add(discriminator_output_layer)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    discriminator_model.compile(optimizer=discriminator_optimizer, loss="binary_crossentropy")

    # train_discriminator_initial(discriminator_model, names, chars)
    gan_model = create_combined_gan(generator_model, discriminator_model)
    n_epochs = 10000
    batch_size = 100
    plot_loss = False
    train_gan(generator_model, discriminator_model, gan_model, names, chars, n_epochs, batch_size, padding_char, plot_loss=plot_loss)

    # show some novel generated data from the generator model
    show_novel_names(generator_model, 1000, padding_char)

    # TODO idea: train generator some by itself using random latent space points paired with actual sample names, so it can learn that it should put those structures in its latent space, more or less; could do some of this each round, or just at the beginning? (I kind of want to do it each round, increase the diversity of the space that the generator knows about by forcing it to keep accommodating new real samples at random latent-space points)

