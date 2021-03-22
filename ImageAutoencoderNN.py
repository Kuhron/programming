# looking at tutorial: https://blog.keras.io/building-autoencoders-in-keras.html

import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import random


encoding_dim = 18  # size of encoded representation (middle layer, the bottleneck)  # 32 in tutorial
input_img = keras.Input(shape=(784,))  # 784 = 28**2, the number of pixels in an MNIST digit image
encoded = layers.Dense(encoding_dim, activation="relu")(input_img)
decoded = layers.Dense(784, activation="sigmoid")(encoded)

autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)  # separate encoder model
encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]  # last layer of the whole autoencoder
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")  # model configuration for how it will optimize and which loss function it will use

(x_train, _), (x_test, _) = mnist.load_data()

# normalizing all values to be in [0,1], and flatten the 28*28 images into vector
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("original data shapes: {} {}".format(x_train.shape, x_test.shape))
x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))
print("new flattened data shapes: {} {}".format(x_train.shape, x_test.shape))

# fit the model, note that the desired output (what would be the y variable) is just the same as the input for an autoencoder (we are measuring loss function about how well the encoded middle layer represents the input data itself once it's passed through the whole network)
epochs = 20  # 50 in tutorial
batch_size = 256  # 256 in tutorial
autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test))
# similarly, the validation data is (input, output) of the test data, but the labels are the same as the input data itself for an autoencoder

# WKJ question: how is training the autoencoder also allowing the separate model objects of encoder and decoder to be trained? are they linked somehow? maybe this happens in how these are all initialized involving each other and/or the same layer objects
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n_digits_to_display = 10
plt.figure(figsize=(20,4))  # in inches
n_digits_in_test_set = x_test.shape[0]
test_indices = random.sample(list(range(n_digits_in_test_set)), n_digits_to_display)
for i in range(n_digits_to_display):
    subplots_rows = 2
    subplots_cols = n_digits_to_display
    im_i = test_indices[i]

    # original MNIST image
    ax = plt.subplot(subplots_rows, subplots_cols, i+1)
    plt.imshow(x_test[im_i].reshape(28,28))  # MNIST image size is 28*28
    plt.gray()  # sets colormap to grayscale
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # image after it's been passed through the autoencoder
    ax = plt.subplot(subplots_rows, subplots_cols, n_digits_to_display + i+1)
    plt.imshow(decoded_imgs[im_i].reshape(28,28))
    plt.gray()  # colormap
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

