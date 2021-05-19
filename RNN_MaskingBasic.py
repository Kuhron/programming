# attempting to get basic masking to work in Keras so I can understand how tf to use it

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import math
import random


SEQ_LEN = 20
PADDING_VAL = 0


def get_sequence():
    n_ints = np.random.randint(3, SEQ_LEN)
    seq = np.random.randint(1, 100, (n_ints,))
    return pad_sequence(seq)


def pad_sequence(seq):
    seq_len, = seq.shape
    n_padding = SEQ_LEN - seq_len
    # pad with no values on the left and n_padding on the right side
    seq = np.pad(seq, pad_width=(0, n_padding), constant_values=PADDING_VAL)
    assert seq.shape == (SEQ_LEN,)
    return seq


def get_value_of_sequence(seq):
    non_padded_seq = seq[seq != PADDING_VAL]
    return np.mean(non_padded_seq)


def create_dataset(n_samples):
    seqs = [get_sequence() for i in range(n_samples)]
    vals = [get_value_of_sequence(seq) for seq in seqs]
    n_timesteps = SEQ_LEN
    n_features = 1
    X = np.array(seqs).reshape(n_samples, n_timesteps, n_features)
    Y = np.array(vals).reshape(n_samples, 1)
    return X, Y


def keras_docs_example():
    # https://keras.io/api/layers/core_layers/masking/
    samples, timesteps, features = 32, 10, 8
    inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
    inputs[:, 3, :] = 0.
    inputs[:, 5, :] = 0.
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0.,
                                      input_shape=(timesteps, features)))
    model.add(tf.keras.layers.LSTM(32))
    
    output = model(inputs)
    # The time step 3 and 5 will be skipped from LSTM calculation.

    print(inputs)
    print(output)

    print("done with keras example")
    # input("press enter to continue")



if __name__ == "__main__":
    keras_docs_example()

    X_train, Y_train = create_dataset(10000)
    X_val, Y_val = create_dataset(500)
    X_test, Y_test = create_dataset(1000)

    n_timesteps = SEQ_LEN
    n_features = 1
    rnn_layer_size = 10
    batch_size = 50
    epochs = 100

    input_shape = (n_timesteps, n_features)
    output_shape = 1  # just a single scalar

    input_layer = layers.Input(input_shape)
    masking_layer = layers.Masking(mask_value=0.0, input_shape=(n_timesteps, n_features))
    rnn_layer = layers.SimpleRNN(rnn_layer_size, return_sequences=False)  # only care about the final output, not sequence
    # ^^^ THIS IS THE ANSWER! rnn layer needs return_sequences=False if loss is caring only about the final output
    # if return_sequences is true, then the loss cares about comparing the SEQUENCE of outputs with a sequence of true outputs, and then you need the output layer to be TimeDistributed(Dense) rather than just Dense

    output_layer = layers.Dense(output_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model = tf.keras.Sequential()

    model.add(input_layer)
    model.add(masking_layer)
    model.add(rnn_layer)
    model.add(output_layer)

    print(model(X_train))  # this WORKS so why does it crash on fitting?
    print(model.summary())

    model.compile(optimizer, loss="mean_squared_error")

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))
    # model.train_on_batch(X_train, Y_train)  # still crashes with "cannot squeeze"
    
