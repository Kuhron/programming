from silence_tensorflow import silence_tensorflow
silence_tensorflow()  # these warnings are super annoying and useless
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import PIL

# based on tutorial at https://www.tensorflow.org/tutorials/generative/dcgan


raise Exception("do not use, too memory-intensive")

def get_array_from_image_fp(fp):
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
    img_fps = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith(".png")]
    # img_fps = random.sample(img_fps, 5)  # debug
    return img_fps


def get_array_from_image_fps(img_fps):
    arrs = []
    for img_fp in img_fps:
        this_arr = get_array_from_image_fp(img_fp)
        arrs.append(this_arr)
    return np.array(arrs)


def get_train_images(image_dir):
    img_fps = get_train_image_fps(image_dir)
    return get_array_from_image_fps(img_fps)


image_dir = "/home/wesley/Desktop/Construction/Conlanging/Cadan Languages/Ilausan/IlausanVineScript/VineScriptCatalog/"
# train_images = get_train_images(image_dir)
img_fps = get_train_image_fps(image_dir)
n_train_images = len(img_fps)

print("got training images")

# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1).astype("float32")
# train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 5
IMAGE_SHAPE = (2220, 1080)
ROWS, COLS = IMAGE_SHAPE

# Batch and shuffle the data
# print("making dataset")
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# print("made dataset")


def make_generator_model():
    # make schedule of how big each layer is, how many filters/strides, etc.
    image_shapes_by_layer = [
        (37, 18),
        # (74, 36),
        (148, 72),
        # (444, 216),
        (2220, 1080),
    ]
    # factors_by_layer = [2, 2, 3, 5]
    factors_by_layer = [148//37, 2220//148]
    # filters_by_layer = [64, 64, 64, 64, 1]
    filters_by_layer = [16, 16, 1]

    model = tf.keras.Sequential()

    # first layer
    model.add(layers.Dense(filters_by_layer[0] * image_shapes_by_layer[0][0] * image_shapes_by_layer[0][1], use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((image_shapes_by_layer[0][0], image_shapes_by_layer[0][1], filters_by_layer[0])))
    assert model.output_shape == (None, image_shapes_by_layer[0][0], image_shapes_by_layer[0][1], filters_by_layer[0])  # Note: None is the batch size

    # subsequent layers with convolution
    for i in range(1, len(image_shapes_by_layer)):
        factor = factors_by_layer[i-1]

        model.add(layers.Conv2DTranspose(filters=filters_by_layer[i], kernel_size=(5, 5), strides=(factor, factor), padding='same', use_bias=False))
        assert model.output_shape == (None, image_shapes_by_layer[i][0], image_shapes_by_layer[i][1], filters_by_layer[i])
        if i < len(image_shapes_by_layer) - 1:
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

    return model


print("making generator model")
input("watch RAM usage")
generator = make_generator_model()
print("made generator model")

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()


def make_discriminator_model():
    # make schedule of how big each layer is, how many filters/strides, etc.
    image_shapes_by_layer = [
        (2220, 1080),
        # (444, 216),
        (148, 72),
        # (74, 36),
        (37, 18),
    ]
    # factors_by_layer = [5, 3, 2, 2]
    factors_by_layer = [2220//148, 148//37]
    # filters_by_layer = [64, 64, 64, 64, 64]
    filters_by_layer = [16, 16, 16]

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=filters_by_layer[0], kernel_size=(5, 5), strides=(factors_by_layer[0], factors_by_layer[0]), padding='same', input_shape=[image_shapes_by_layer[0][0], image_shapes_by_layer[0][1], 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    for i in range(len(image_shapes_by_layer)):
        factor = factors_by_layer[i-1]
        model.add(layers.Conv2D(filters=filters_by_layer[i], kernel_size=(5, 5), strides=(factor, factor), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model



print("making discriminator model")
discriminator = make_discriminator_model()
print("made discriminator model")
decision = discriminator(generated_image)
print("discriminator's decision about the previously shown random noise image:", decision)


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
print("checkpoint dir declared")


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(image_fps_in_batch):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    print("made noise", noise)

    images = get_array_from_image_fps(image_fps_in_batch)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        print(f"{gen_loss = }, {disc_loss = }")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(img_fps, epochs):
    for epoch in range(epochs):
        print(f"{epoch = }")
        start = time.time()

        # batch up the fps myself
        fp_batches = []
        seen = set()
        for i in range(n_train_images // BATCH_SIZE):
            batch = random.sample([x for x in img_fps if x not in seen], BATCH_SIZE)
            seen |= set(batch)
            fp_batches.append(batch)

        for batch_i, fp_batch in enumerate(fp_batches):
            print(f"new image batch ({batch_i+1} of {len(fp_batches)}) at time {time.time()}")
            train_step(fp_batch)

        # Produce images as you go
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             seed)


OUTPUT_IMAGE_DIR = "Images/VineScriptGAN/"

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_IMAGE_DIR, f"image_at_epoch_{epoch}-IMG{i}.png"))
        plt.gcf().clear()


print("training")
train(img_fps, EPOCHS)
print("done training")


# in case need to restore a checkpoint reached during earlier training, e.g. if the training was interrupted
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


