from silence_tensorflow import silence_tensorflow
silence_tensorflow()  # these warnings are super annoying and useless
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.utils.generic_utils import get_custom_objects

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import PIL
import glob

from TabletInput import get_array_from_data_fp, draw_glyph, plot_time_series, write_data_to_file, MIN_PRESSURE_FOR_STROKE



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
    n_cols = 4
    all_dts = set()
    for fp in training_data_fps:
        l = get_array_from_data_fp(fp, binarize_pressure_threshold=MIN_PRESSURE_FOR_STROKE)
        # plot_time_series(l)  # debug
        assert ((l[:, -1] == 0) | (l[:, -1] == 1)).all(), f"training data should have binary pressure; problem fp is {fp}"
        zero_p_indices, = np.where(l[:, -1] == 0)
        assert zero_p_indices[-1] == l.shape[0] - 1
        ts = l[:, 0]
        dts = np.diff(ts)

        # now add interpolation zones for when the pen is up, with pressure at zero and x and y going from the last touch point to the first of the next stroke
        # this will hopefully help the network learn better since it won't be a discontinuous jump
        dts_at_nonzero_pressure = dts[np.where(l[:, -1] != 0)[0]]
        assert (dts_at_nonzero_pressure < 100).all(), f"unusual dt values: {sorted(set(dts_at_nonzero_pressure))}"
        mean_dt = np.mean(dts_at_nonzero_pressure)

        stroke_arrs = [l[:zero_p_indices[0]]]  # there must be at least one zero p, at the very end of the glyph
        for ii in range(len(zero_p_indices) - 1):
            i = zero_p_indices[ii]
            t0, x0, y0, p0 = l[i]
            # fill the gap after this with some fake points that have zero pressure and interpolate x and y until the next stroke starts
            dt = dts[i]  # time from i to i+1
            n_ticks = int(round(dt / mean_dt))
            if n_ticks > 0:
                dx = l[i+1, 1] - l[i, 1]
                dy = l[i+1, 2] - l[i, 2]
                dt_per_tick = dt / n_ticks
                dx_per_tick = dx / n_ticks
                dy_per_tick = dy / n_ticks
                gap_arr = []
                for j in range(1, n_ticks):  # already have the j=0 point which is the zero-pressure point, don't want to duplicate it
                    t = t0 + j*dt_per_tick
                    x = x0 + j*dx_per_tick
                    y = y0 + j*dy_per_tick
                    p = 0
                    gap_arr.append([t, x, y, p])
                if len(gap_arr) > 0:
                    gap_arr = np.array(gap_arr)
                    stroke_arrs.append(gap_arr)
            next_stroke_arr = l[i+1 : zero_p_indices[ii+1] + 1]
            stroke_arrs.append(next_stroke_arr)
        l = np.concatenate(stroke_arrs, axis=0)
        # plot_time_series(l)  # debug
        # draw_glyph(l, pressure_threshold=0.5)  # debug
        r, c = l.shape
        assert c == n_cols
        arrs.append(l)

    max_n_rows = max(l.shape[0] for l in arrs)

    # once have the time padding between strokes figured out, get rid of timestamps so network doesn't have to learn them
    arrs = [l[:, 1:] for l in arrs]
    # now pad arrays with empty data
    # or can try expanding them (like "justify margins") or something else to normalize shape
    arrs = [pad_with_zeros(a, max_n_rows, xy_val=0.5) for a in arrs]
    arrs = np.array(arrs)
    return arrs


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



# image_dir = "/home/wesley/Desktop/Construction/Conlanging/Cadan Languages/Ilausan/IlausanVineScript/VineScriptCatalog/"
training_data_dir = "VineScriptTabletInputData"
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


def make_generator_model():
    model = tf.keras.Sequential()

    n_dense_neurons = 100
    n_filters = 64

    model.add(layers.Dense(n_dense_neurons * n_cols, input_shape=(100,)))  # first layer connected to input from latent space
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(n_dense_neurons * n_cols))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    # DON'T want activation on output layer, want it to be able to return whatever values it wants in the array
    # (could cause problems trying to get zero pressure though)
    # ideal would be activate x and y with sigmoid to keep them inside the image,
    # and activate pressure with ReLU or some kind of half-sigmoid that is zero for input <=0 and goes up to 1
    # can I make a custom activation function that will behave differently on the different columns like this?
    # then can train on x and y rather than dx and dy (because the latter has too much cumulative drift making glyphs that go way out of bounds)

    # try 1D convolution to capture the time series nature where adjacent points are correlated
    # model.add(layers.Dense(n_ticks * n_cols))  # so there will be the correct number of points for deconvolving into time series of the correct number of ticks and channels
    model.add(layers.Reshape((n_dense_neurons, n_cols)))  # there are n_cols channels in the data
    model.add(layers.Conv1DTranspose(filters=n_filters, kernel_size=5, padding="same"))

    # not sure why convolution doesn't operate separately on the channels; is it convolving them together??? so I guess I'll just make another freaking dense layer to get the shape correct again
    model.add(layers.Flatten())
    model.add(layers.Dense((n_ticks * n_cols)))
    model.add(layers.Reshape((n_ticks, n_cols)))

    # model.add(layers.Activation(bowl_activation))
    model.add(layers.Activation("sigmoid"))

    expected_output_shape = (None, n_ticks, n_cols)  # first element is batch size, can be None
    assert model.output_shape == expected_output_shape, f"expected model output shape {expected_output_shape} but got {model.output_shape}"

    return model


print("making generator model")
input("watch RAM usage")
generator = make_generator_model()
print("made generator model")

noise = tf.random.normal([1, 100])
generated_arr = generator(noise, training=False).numpy()
assert generated_arr.shape[0] == 1  # 1 sample
# plot_time_series(generated_arr[0])
# draw_glyph(generated_arr[0])


def make_discriminator_model():
    n_dense_neurons = 100
    n_filters = 64

    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(n_ticks, n_cols)))
    # model.add(layers.Flatten())
    model.add(layers.BatchNormalization())

    # model.add(layers.Reshape((n_ticks, n_cols)))  # there are n_cols channels in the data and n_ticks timesteps
    model.add(layers.Conv1D(filters=n_filters, kernel_size=5, padding="same"))
    model.add(layers.Flatten())

    model.add(layers.Dense(n_dense_neurons * n_cols))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(n_dense_neurons * n_cols))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1, activation="sigmoid"))  # want logistic regression-like output
    assert model.output_shape == (None, 1), model.output_shape

    return model



print("making discriminator model")
discriminator = make_discriminator_model()
print("made discriminator model")
decision = discriminator(generated_arr)
print("discriminator's decision about the previously shown random noise image:", decision)


# def tf_print(x, message=None):
#     if message is None:
#         message = "values: "
#     x = tf.Print(x, [x], message=message)
#     return x


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './NeuralNetFiles/VineScriptGAN_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
print("checkpoint dir declared")


EPOCHS = 100000
noise_dim = 100
num_examples_to_generate = 5

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function  # seems to do weird stuff like make random.random() always be the same! don't like this
def train_step(batch):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)
        # print("generated data:", generated_data)

        real_output = discriminator(batch, training=True)
        # print("real output:", real_output)
        fake_output = discriminator(generated_data, training=True)
        # print("fake output:", fake_output)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    print(f"gen_loss = {gen_loss.numpy()}, disc_loss = {disc_loss.numpy()}")
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # print("gen grad:", gradients_of_generator)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # print("disc grad:", gradients_of_discriminator)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"{epoch = }")
        start = time.time()

        for batch in dataset:
            train_step(batch)

        # Produce images as you go
        if epoch == 0 or (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

        # Save the model
        if (epoch + 1) % 100 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("saved checkpoint")

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)


OUTPUT_DIR = "Images/VineScriptGAN/"

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False).numpy()
    print(f"{predictions.shape = }")
    for i in range(predictions.shape[0]):
        l = predictions[i]
        tsv_fp = os.path.join(OUTPUT_DIR, f"E{epoch}_testinput{i}_Array.tsv")
        write_data_to_file(l, tsv_fp)
        plot_time_series(l, show=False)
        time_series_fp = os.path.join(OUTPUT_DIR, f"E{epoch}_testinput{i}_TimeSeries.png")
        plt.savefig(time_series_fp)
        plt.gcf().clear()
        draw_glyph(l, pressure_threshold=0.5, show=False)
        glyph_fp = os.path.join(OUTPUT_DIR, f"E{epoch}_testinput{i}_Glyph.png")
        plt.savefig(glyph_fp)
        plt.gcf().clear()
    print("generated images")

print("training")
train(train_dataset, EPOCHS)
print("done training")

# in case need to restore a checkpoint reached during earlier training, e.g. if the training was interrupted
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# TODO make sure pressure can be output as exactly zero by the model, so the pen can be picked up (might need ReLU or some other activation like that on the last layer for this to be possible)

