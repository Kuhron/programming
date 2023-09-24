from silence_tensorflow import silence_tensorflow
silence_tensorflow()  # these warnings are super annoying and useless
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import PIL
import glob

from TabletInput import get_array_from_data_fp, draw_glyph, plot_time_series, write_data_to_file, MIN_PRESSURE_FOR_STROKE
from VineScriptGAN import get_training_data_fps, get_training_data_array



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # https://keras.io/examples/timeseries/timeseries_transformer_classification/
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


if __name__ == "__main__":
    training_data_dir = "VineScriptTabletInputData"
    OUTPUT_DIR = "Images/VineScriptGAN/"
    training_data_fps = get_training_data_fps(training_data_dir)
    n_train = len(training_data_fps)
    train_arr = get_training_data_array(training_data_fps)
    assert train_arr.shape[0] == n_train
    n_ticks = train_arr.shape[1]
    n_cols = 3  # x, y, pressure
    assert train_arr.shape[2] == n_cols

    print(f"got training data, of shape {train_arr.shape}")

    BUFFER_SIZE = 60000
    BATCH_SIZE = 72

    # Batch and shuffle the data
    print("making dataset")
    train_dataset = tf.data.Dataset.from_tensor_slices(train_arr).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print("made dataset")

    input_shape = train_arr.shape[1:]

    model = build_transformer_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )

    model.evaluate(x_test, y_test, verbose=1)

