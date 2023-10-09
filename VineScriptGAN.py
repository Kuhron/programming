from silence_tensorflow import silence_tensorflow
silence_tensorflow()  # these warnings are super annoying and useless
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers
from scipy.interpolate import CubicSpline

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import PIL
import glob
from datetime import datetime

from TabletInput import get_array_from_data_fp, draw_glyph_from_xyp_time_series, draw_glyph_from_simultaneous_strokes, plot_xyp_time_series, plot_simultaneous_strokes, write_data_to_file_from_xyp_time_series, write_data_to_file_from_simultaneous_strokes, MIN_PRESSURE_FOR_STROKE



def get_array_from_image_fp(fp):
    raise Exception("do not use, too memory-intensive")
    im = PIL.Image.open(fp)
    im_rgb = im.convert("RGB")
    a = np.array(im_rgb)
    x,y,z = a.shape
    a = np.mean(a, axis=2)  # grayscale, just treat the overall brightness of the pixel as its 1-dimensional value
    x2, y2 = a.shape
    assert x2 == x and y2 == y
    print(f"got array for image at {fp}")
    return a


def get_train_image_fps(image_dir):
    raise Exception("do not use, too memory-intensive")
    img_fps = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith(".png")]
    # img_fps = random.sample(img_fps, 5)  # debug
    return img_fps


def get_array_from_image_fps(img_fps):
    raise Exception("do not use, too memory-intensive")
    arrs = []
    for img_fp in img_fps:
        this_arr = get_array_from_image_fp(img_fp)
        arrs.append(this_arr)
    return np.array(arrs)


def get_train_images(image_dir):
    raise Exception("do not use, too memory-intensive")
    img_fps = get_train_image_fps(image_dir)
    return get_array_from_image_fps(img_fps)


def get_training_data_fps(parent_dir):
    return glob.glob(f"{parent_dir}/**/*.tsv")


def get_training_data_array(training_data_fps):
    arrs = []

    raw_arrs_per_glyph = []
    for fp in training_data_fps:
        l = get_array_from_data_fp(fp, binarize_pressure_threshold=MIN_PRESSURE_FOR_STROKE)
        raw_arrs_per_glyph.append(l)

    stroke_arrs_per_glyph = []
    for l in raw_arrs_per_glyph:
        ts = l[:, 0]

        # plot_xyp_time_series(l)  # debug
        # draw_glyph_from_xyp_time_series(l, pressure_threshold=0.5)  # debug
        assert ((l[:, -1] == 0) | (l[:, -1] == 1)).all(), f"training data should have binary pressure; problem fp is {fp}"
        zero_p_indices, = np.where(l[:, -1] == 0)
        assert zero_p_indices[-1] == l.shape[0] - 1

        stroke_arrs_this_glyph = [l[:zero_p_indices[0]+1]]  # there must be at least one zero p, at the very end of the glyph
        for ii in range(len(zero_p_indices) - 1):
            i = zero_p_indices[ii]
            next_stroke_arr = l[i+1 : zero_p_indices[ii+1] + 1]
            stroke_arrs_this_glyph.append(next_stroke_arr)
        stroke_arrs_per_glyph.append(stroke_arrs_this_glyph)
    max_n_time_points = max(max(stroke_arr.shape[0] for stroke_arr in stroke_arrs_this_glyph) for stroke_arrs_this_glyph in stroke_arrs_per_glyph)
    max_n_strokes = max(len(stroke_arrs_this_glyph) for stroke_arrs_this_glyph in stroke_arrs_per_glyph)
    # print(f"{max_n_time_points = }, {max_n_strokes = }")

    n_strokes_vector_per_glyph = []
    for stroke_arrs in stroke_arrs_per_glyph:
        stroke_arrs, n_strokes_in_glyph = normalize_stroke_arrays_of_glyph(stroke_arrs, desired_n_time_points=max_n_time_points, desired_n_strokes=max_n_strokes)
        l = np.stack(stroke_arrs)
        # plot_simultaneous_strokes(l)  # debug
        # draw_glyph_from_simultaneous_strokes(l)  # debug
        n_strokes_raw, n_time_points, n_channels = l.shape
        assert n_strokes_raw == max_n_strokes
        assert n_channels == 2  # x and y
        assert n_time_points == max_n_time_points
        arrs.append(l)
        n_strokes_vector = [int(i == n_strokes_in_glyph) for i in range(1, max_n_strokes+1)]
        n_strokes_vector_per_glyph.append(n_strokes_vector)

    xy_arr = np.stack(arrs)
    n_strokes_arr = np.array(n_strokes_vector_per_glyph)
    return xy_arr, n_strokes_arr


def normalize_stroke_arrays_of_glyph(stroke_arrs, desired_n_time_points, desired_n_strokes):
    # make all strokes the same length in time
    # treat all strokes as happening simultaneously in different channels
    # pad with zeros for extra stroke slots when glyph has less than max number of strokes
    # always start first stroke at (x,y) = (0,0)
    # remove pressure completely
    # scale xs and ys to be standard deviations away from origin

    xs_per_stroke = []
    ys_per_stroke = []
    for stroke_arr in stroke_arrs:
        ts = stroke_arr[:, 0]
        xs = stroke_arr[:, 1]
        ys = stroke_arr[:, 2]
        ps = stroke_arr[:, 3]
        assert ps[-1] == 0 and (ps[:-1] == 1).all(), ps
        # now ignore time and pressure completely
        xs_per_stroke.append(xs)
        ys_per_stroke.append(ys)
    xs_this_glyph = np.concatenate(xs_per_stroke)
    ys_this_glyph = np.concatenate(ys_per_stroke)
    assert len(xs_this_glyph.shape) == len(ys_this_glyph.shape) == 1
    assert xs_this_glyph.shape == ys_this_glyph.shape
    x_std = np.std(xs_this_glyph)
    y_std = np.std(ys_this_glyph)
    std = (x_std * y_std) ** 0.5  # geometric mean so aspect ratio is same (I want to preserve aspect ratio, maybe a bad idea? idk, we'll see)
    x0 = stroke_arrs[0][0,1]
    y0 = stroke_arrs[0][0,2]

    new_xs_per_stroke = []
    new_ys_per_stroke = []
    for xs, ys in zip(xs_per_stroke, ys_per_stroke):
        # first scale in space
        xs = [(x-x0)/std for x in xs]
        ys = [(y-y0)/std for y in ys]

        # then dilate time
        assert len(xs) == len(ys)
        assert len(xs) <= desired_n_time_points
        raw_n_time_points = len(xs)
        ts_raw = np.linspace(0, 1, raw_n_time_points)

        use_spline = False  # spline ends up looking too wobbly
        ts_desired = np.linspace(0, 1, desired_n_time_points)
        if use_spline:
            xs_spl = CubicSpline(ts_raw, xs)
            ys_spl = CubicSpline(ts_raw, ys)
            new_xs = xs_spl(ts_desired)
            new_ys = ys_spl(ts_desired)
        else:
            new_xs = np.interp(ts_desired, ts_raw, xs)
            new_ys = np.interp(ts_desired, ts_raw, ys)

        assert len(new_xs) == len(new_ys) == desired_n_time_points
        assert np.isclose(new_xs[0], xs[0], rtol=1e-8), f"{new_xs[0] = } != {xs[0] = }"
        assert np.isclose(new_xs[-1], xs[-1], rtol=1e-8), f"{new_xs[-1] = } != {xs[-1] = }"
        assert np.isclose(new_ys[0], ys[0], rtol=1e-8), f"{new_ys[0] = } != {ys[0] = }"
        assert np.isclose(new_ys[-1], ys[-1], rtol=1e-8), f"{new_ys[-1] = } != {ys[-1] = }"
        new_xs_per_stroke.append(new_xs)
        new_ys_per_stroke.append(new_ys)

        # plt.plot(new_xs, new_ys)

    # plt.axis("equal")
    # plt.show()

    n_strokes_in_glyph = len(stroke_arrs)
    n_strokes_left = desired_n_strokes - n_strokes_in_glyph
    assert n_strokes_left >= 0
    for i in range(n_strokes_left):
        xs = [0] * desired_n_time_points
        ys = [0] * desired_n_time_points
        new_xs_per_stroke.append(xs)
        new_ys_per_stroke.append(ys)
    new_stroke_arrs = []
    for i in range(desired_n_strokes):
        new_stroke_arr = np.stack([new_xs_per_stroke[i], new_ys_per_stroke[i]], axis=-1)
        new_stroke_arrs.append(new_stroke_arr)
    return new_stroke_arrs, n_strokes_in_glyph


def pad_with_zeros(arr, n_rows, xy_val=0):
    r, c = arr.shape
    zeros = np.zeros((n_rows - r, c))
    # if the first two columns are x and y coords, use some default value (e.g. 0.5 for the middle of the box)
    zeros[:,0] = xy_val
    zeros[:,1] = xy_val
    new_arr = np.concatenate([arr, zeros], axis=0)
    assert new_arr.shape == (n_rows, c)
    return new_arr


@tf.function
def sigmoid(x):
    return 1/(1 + tf.math.exp(-x))


@tf.function
def softplus(x):
    return tf.math.log(1 + tf.math.exp(x))


@tf.function
def inverse_softplus(x):
    return tf.where(x <= 0, 0*x, tf.math.log(tf.math.exp(x) - 1))


@tf.function
def half_sigmoid(x):
    e = np.exp(1)
    # return sigmoid(e * inverse_softplus(x))
    x = tf.where(x <= 0, 1+0*x, x)  # try hacking so no negative x is passed ANYWHERE into the sigmoid stuff (because it seems that a single NaN produced will pollute the whole array with NaNs)
    return 1/(1+ (tf.math.exp(x) - 1)**(-e))  # simplified algebra to try to avoid NaN for x >= 0; might choose an exponent like -2 rather than -e so that this is still defined at zero?


@tf.function
def sigmoid_in_01_box(x):
    a = 2
    c = 0.5
    epsilon = 1e-16
    return 1/(1+((1/(x+epsilon)-1)**a)*(1/c-1))


@tf.function
def half_sigmoid_activation(x):
    # avoid NaN by making all branches of all .where/.cond functions always differentiable and evaluable everywhere, because tf will evaluate both branches and propagate NaNs up even if they are in a branch that is not chosen

    return tf.where(x <= 0, 0*x, tf.where(x >= 1, 1 + 0*x, sigmoid_in_01_box(x)))

    # true_fn = lambda: 0*x  # so it will have the same "overall structure of return values" (a tensor of zeros or whatever it wants)
    # false_fn = lambda: half_sigmoid(x)
    # return tf.cond(x <= 0, true_fn, false_fn)

    # max_0_x = keras.activations.relu(x)
    # return half_sigmoid(max_0_x)


@tf.function
def dip_activation(x):
    # so things can get to or very close to zero with little effort (e.g. the pressure)
    # make exponent a positive even integer, bigger for more flatness around zero and steeper sigmoid on either side
    return 1 - 1/(1+x**4)


@tf.function
def bowl_activation(x):
    return 1/2 * (softplus(x) + softplus(-x)) - softplus(0*x)


# @tf.function
# def truncate_low_pressure(x):
#     # so that pressure that's low enough from generator sigmoid can be set to exactly zero,
#     # with minimal impact on x and y coordinates passed through this same function (since I basically never draw so close to the edge)
#     return tf.where(x < MIN_PRESSURE_FOR_STROKE, 0*x, x)


def make_generator_model(n_strokes, n_time_points, n_channels):
    n_dense_neurons = 50
    n_filters = 16

    # going to use the functional API of Keras to be able to handle multiple inputs and outputs (one for the simultaneous-strokes array, and one for the small vector just saying how many strokes this glyph has)

    inputs = keras.Input(shape=(noise_dim,))
    mid = inputs
    # mid = layers.BatchNormalization()(mid)
    mid = layers.Dense(n_dense_neurons * n_channels, activation=layers.LeakyReLU())(mid)
    # mid = layers.Dropout(0.2)(mid)

    # mid = layers.GaussianNoise(0.001)(mid)  # trying to prevent model from getting stuck in rut
    mid = layers.Dense(n_dense_neurons * n_channels, activation=layers.LeakyReLU())(mid)
    # mid = layers.Dropout(0.2)(mid)

    # try 1D convolution to capture the time series nature where adjacent points are correlated
    mid = layers.Reshape((n_dense_neurons * n_channels, 1))(mid)

    conv1 = layers.Conv1DTranspose(filters=n_filters, kernel_size=9, strides=2, padding="same")(mid)
    conv1 = layers.Flatten()(conv1)

    conv2 = layers.Conv1DTranspose(filters=n_filters, kernel_size=23, strides=3, padding="same")(mid)
    conv2 = layers.Flatten()(conv2)

    conv3 = layers.Conv1DTranspose(filters=n_filters, kernel_size=64, strides=7, padding="same")(mid)
    conv3 = layers.Flatten()(conv3)

    mid = layers.concatenate([conv1, conv2, conv3])
    mid = layers.Dense(n_strokes * n_time_points * n_channels)(mid)  # DON'T want activation on the last layer since we want it to be able to take various positive and negative values

    # from here now we make the simultaneous-strokes array (xs and ys), and also the vector saying how many strokes to use
    out_xys = layers.Reshape((n_strokes, n_time_points, n_channels))(mid)
    out_n_strokes = layers.Dense(n_strokes, activation="softmax")(mid)

    model = keras.Model(inputs=inputs, outputs=[out_xys, out_n_strokes])
    expected_output_shape = [(None, n_strokes, n_time_points, n_channels), (None, n_strokes)]  # first element is batch size, can be None
    assert model.output_shape == expected_output_shape, f"expected model output shape {expected_output_shape} but got {model.output_shape}"

    return model


def make_discriminator_model(n_strokes, n_time_points, n_channels):
    n_dense_neurons = 50
    n_filters = 16

    xy_inputs = keras.Input(shape=(n_strokes, n_time_points, n_channels))
    xy_mid = xy_inputs
    n_strokes_inputs = keras.Input(shape=(n_strokes,))  # greatest value in this vector tells how many strokes to pay attention to (first n)
    n_strokes_mid = n_strokes_inputs

    # xy_mid = layers.BatchNormalization()(xy_mid)

    conv1 = layers.Conv1D(filters=n_filters, kernel_size=9, strides=2, padding="same")(xy_mid)
    conv1 = layers.Flatten()(conv1)

    conv2 = layers.Conv1D(filters=n_filters, kernel_size=19, strides=3, padding="same")(xy_mid)
    conv2 = layers.Flatten()(conv2)

    conv3 = layers.Conv1D(filters=n_filters, kernel_size=119, strides=16, padding="same")(xy_mid)
    conv3 = layers.Flatten()(conv3)

    xy_mid = layers.concatenate([conv1, conv2, conv3])

    n_strokes_mid = layers.Flatten()(n_strokes_mid)

    mid = layers.concatenate([xy_mid, n_strokes_mid])
    mid = layers.Dense(n_dense_neurons * n_channels, activation=layers.LeakyReLU())(mid)
    # mid = layers.Dropout(0.2)(mid)

    # mid = layers.GaussianNoise(0.01)(mid)  # trying to prevent model from getting stuck in rut

    mid = layers.Dense(n_dense_neurons * n_channels, activation=layers.LeakyReLU())(mid)
    # mid = layers.Dropout(0.2)(mid)

    out = layers.Dense(1, activation="sigmoid")(mid)  # want logistic regression-like output
    model = keras.Model(inputs=[xy_inputs, n_strokes_inputs], outputs=[out])
    assert model.output_shape == (None, 1), model.output_shape

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def print_layers(model, x_train):
    # print("Model summary:")
    # print(model.summary())
    # print()
    x = x_train[:1]
    for i in range(len(model.layers)):
        print(f"Layer {i}: {model.layers[i]}")
        new_model = keras.models.Sequential(model.layers[:i+1])
        # print("New model summary:")
        # print(new_model.summary())
        # print()
        # print("output as of this layer:")
        print(new_model(x, training=True))  # need training=True to make the GaussianNoise layer apply
        print()


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function  # seems to do weird stuff like make random.random() always be the same! don't like this
def train_step(batch, epoch):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    batch_xys = batch["xy"]
    batch_n_strokes = batch["n_strokes"]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_xys, generated_n_strokes = generator(noise, training=True)

        real_output = discriminator([batch_xys, batch_n_strokes], training=True)
        # print("real output:", real_output)
        fake_output = discriminator([generated_xys, generated_n_strokes], training=True)
        # print("fake output:", fake_output)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_over_disc_loss_min = 0.1
    disc_over_gen_loss_min = 0.1
    # for each model, if the loss is too low, don't train it
    train_gen = gen_loss / disc_loss >= gen_over_disc_loss_min
    train_disc = disc_loss / gen_loss >= disc_over_gen_loss_min

    if train_gen:
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    if train_disc:
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    train_str = (["gen"] if train_gen else []) + (["disc"] if train_disc else [])
    s = f"gen_loss = {gen_loss.numpy():.6f}, disc_loss = {disc_loss.numpy():.6f}, training: {train_str}"
    print(s)

    # print("\nGenerator:")
    # print_layers(generator, noise)
    # print("Discriminator:")
    # print_layers(discriminator, batch)


def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"{epoch = }")
        start = time.time()

        for batch in dataset:
            train_step(batch, epoch)

        # Produce images as you go
        if epoch == 0 or (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, test_input=None)

        # Save the model
        if (epoch + 1) % 25 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("saved checkpoint")

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, test_input=None)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    if test_input is None:
        test_input = tf.random.normal([num_examples_to_generate, noise_dim])
    predictions, n_strokes_vector = model(test_input, training=False)
    predictions = predictions.numpy()
    n_strokes_vector = n_strokes_vector.numpy()
    print(f"{predictions.shape = }")
    now_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    for i in range(predictions.shape[0]):
        l = predictions[i]
        n_strokes_this_glyph = 1 + np.argmax(n_strokes_vector[i])
        fp_prefix = f"{now_str}_E{epoch}_testinput{i}_"
        tsv_fp = os.path.join(OUTPUT_DIR, fp_prefix + "Array.tsv")
        write_data_to_file_from_simultaneous_strokes(l, tsv_fp)
        plot_simultaneous_strokes(l, n_strokes_this_glyph, show=False)
        time_series_fp = os.path.join(OUTPUT_DIR, fp_prefix + "TimeSeries.png")
        plt.savefig(time_series_fp)
        plt.gcf().clear()
        draw_glyph_from_simultaneous_strokes(l, n_strokes_this_glyph, show=False)
        glyph_fp = os.path.join(OUTPUT_DIR, fp_prefix + "Glyph.png")
        plt.savefig(glyph_fp)
        plt.gcf().clear()
    print("generated images")


def get_xy_marks(xs, ys, ps, pressure_threshold):
    # averages and other stats about xs and ys where p is not zero
    xs = xs[ps >= pressure_threshold]
    ys = ys[ps >= pressure_threshold]
    n_std = 1

    x_mean = np.mean(xs)
    x_min = min(xs)
    x_max = max(xs)
    x_std = np.std(xs)
    x_low = x_mean - n_std * x_std
    x_high = x_mean + n_std * x_std

    y_mean = np.mean(ys)
    y_min = min(ys)
    y_max = max(ys)
    y_std = np.std(ys)
    y_low = y_mean - n_std * y_std
    y_high = y_mean + n_std * y_std

    return x_min, x_low, x_mean, x_high, x_max, y_min, y_low, y_mean, y_high, y_max


def draw_glyphs_with_xy_marks():
    indices = list(range(n_train))
    random.shuffle(indices)
    for i in indices:
        row = train_arr[i]
        xs = row[:, 0]
        ys = row[:, 1]
        ps = row[:, 2]
        x_min, x_low, x_mean, x_high, x_max, y_min, y_low, y_mean, y_high, y_max = get_xy_marks(xs, ys, ps, 0.5)
        draw_glyph_from_xyp_time_series(row, 0.5, show=False)

        plt.plot([x_min, x_min], [y_min, y_max], c="g")
        plt.plot([x_max, x_max], [y_min, y_max], c="g")
        plt.plot([x_min, x_max], [y_min, y_min], c="g")
        plt.plot([x_min, x_max], [y_max, y_max], c="g")
        plt.plot([x_low, x_low], [y_low, y_high], c="y")
        plt.plot([x_high, x_high], [y_low, y_high], c="y")
        plt.plot([x_low, x_high], [y_low, y_low], c="y")
        plt.plot([x_low, x_high], [y_high, y_high], c="y")
        plt.plot([x_mean, x_mean], [y_low, y_high], c="r")
        plt.plot([x_low, x_high], [y_mean, y_mean], c="r")

        plt.show()


def show_example_of_model_output():
    noise = tf.random.normal([1, noise_dim])
    generated_arr, generated_n_strokes_vector = generator(noise, training=False)
    generated_arr = generated_arr.numpy()
    generated_n_strokes_vector = generated_n_strokes_vector.numpy()
    assert generated_arr.shape[0] == 1  # 1 sample
    plot_simultaneous_strokes(generated_arr[0], n_strokes = 1 + np.argmax(generated_n_strokes_vector[0]))
    draw_glyph_from_simultaneous_strokes(generated_arr[0], n_strokes = 1 + np.argmax(generated_n_strokes_vector[0]))
    decision = discriminator([generated_arr, generated_n_strokes_vector])
    print("discriminator's decision about the previously shown random noise image:", decision)



if __name__ == "__main__":
    # image_dir = "/home/wesley/Desktop/Construction/Conlanging/Cadan Languages/Ilausan/IlausanVineScript/VineScriptCatalog/"
    training_data_dir = "VineScriptTabletInputData"
    OUTPUT_DIR = "Images/VineScriptGAN/"
    training_data_fps = get_training_data_fps(training_data_dir)
    random.shuffle(training_data_fps)
    n_train = len(training_data_fps)
    train_xy_arr, train_n_strokes_arr = get_training_data_array(training_data_fps)
    assert len(train_xy_arr.shape) == 4, train_xy_arr.shape
    assert train_xy_arr.shape[0] == n_train, train_xy_arr.shape
    max_n_strokes = train_xy_arr.shape[1]
    n_time_points = train_xy_arr.shape[2]
    n_channels = train_xy_arr.shape[3]
    assert n_channels == 2  # x and y

    print(f"got training data, of shape {train_xy_arr.shape} for xys and {train_n_strokes_arr.shape} for n_strokes")

    # trying to understand what the convolutions look like
    for i in range(4):
        sub_arr = train_xy_arr[i]
        sub_arr_n_strokes = 1 + np.argmax(train_n_strokes_arr[i])
        plot_simultaneous_strokes(sub_arr, sub_arr_n_strokes)
        draw_glyph_from_simultaneous_strokes(sub_arr, sub_arr_n_strokes)

    # draw_glyphs_with_xy_marks()  # to see e.g. what the min/max width are, or the standard deviation of height and width, drawn as lines on the glyph, for understanding of how to scale the data

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    noise_dim = 100

    # Batch and shuffle the data
    print("making dataset")
    train_dataset = tf.data.Dataset.from_tensor_slices({"xy": train_xy_arr, "n_strokes": train_n_strokes_arr}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print("made dataset")

    print("making generator model")
    input("watch RAM usage")
    generator = make_generator_model(max_n_strokes, n_time_points, n_channels)
    print("made generator model")

    print("making discriminator model")
    discriminator = make_discriminator_model(max_n_strokes, n_time_points, n_channels)
    print("made discriminator model")

    keras.utils.plot_model(generator, to_file=os.path.join(OUTPUT_DIR, "model_generator.png"), show_shapes=True)
    keras.utils.plot_model(discriminator, to_file=os.path.join(OUTPUT_DIR, "model_discriminator.png"), show_shapes=True)

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './NeuralNetFiles/VineScriptGAN_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    print("checkpoint dir declared")

    print("restoring checkpoint")
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("restored latest checkpoint")
    show_example_of_model_output()

    EPOCHS = 100000
    num_examples_to_generate = 5

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    print("training")
    train(train_dataset, EPOCHS)
    print("done training")

    # in case need to restore a checkpoint reached during earlier training, e.g. if the training was interrupted

