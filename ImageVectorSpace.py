# playing with trying to make a vector space of images (maybe using PCA)
# and then looking at random points in the space to see what they look like
# inspired by those animations where they morph continuously from one image to another
# (presumably following a straight line through the vector space from one data point to another)


import os
import time
from datetime import datetime
from glob import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

import ImageDatasetLoadPhonePhotos as phonephotos



def get_average_of_images(n_images_to_average, n_training_images, X_pca):
    images_to_average = random.sample(list(range(n_training_images)), n_images_to_average)
    new_image_X_pca = np.zeros((X_pca.shape[1],))
    for image_i in images_to_average:
        vector = X_pca[image_i, :]
        assert vector.shape == new_image_X_pca.shape
        new_image_X_pca += vector
    new_image_X_pca /= n_images_to_average
    return new_image_X_pca


def get_random_image_in_pca_space(n_components, X_pca):
    # another approach: choose a random alpha along the range of each component
    alphas = np.random.uniform(0, 1, (n_components,))
    X_pca_min_by_component, X_pca_max_by_component = get_X_pca_min_max_by_component(X_pca)
    new_image_X_pca = X_pca_min_by_component + alphas * (X_pca_max_by_component - X_pca_min_by_component)
    return new_image_X_pca


def get_X_pca_min_max_by_component(X_pca):
    X_pca_max_by_component = X_pca.max(axis=0)  # gets max value for each component among the data points in X_pca
    X_pca_min_by_component = X_pca.min(axis=0)  # gets max value for each component among the data points in X_pca
    return X_pca_min_by_component, X_pca_max_by_component


def get_X_pca_max_abs_by_component(X_pca):
    return abs(X_pca).max(axis=0)


def fix_clipping(image_arr):
    # puts the array in rgb range 0 1 for each color, so matplotlib wont "clip" the out-of-range values
    # you can get out of range when you make a new point in PCA space and then convert it back to RGB
    assert len(image_arr.shape) == 3, image_arr.shape
    assert image_arr.shape[-1] == 3, image_arr.shape  # the last dimension is where the RGB values live
    reds = image_arr[:,:,0]
    greens = image_arr[:,:,1]
    blues = image_arr[:,:,2]
    min_red = reds.min()
    max_red = reds.max()
    min_green = greens.min()
    max_green = greens.max()
    min_blue = blues.min()
    max_blue = blues.max()
    reds_01 = (reds - min_red) / (max_red - min_red)
    greens_01 = (greens - min_green) / (max_green - min_green)
    blues_01 = (blues - min_blue) / (max_blue - min_blue)
    image_arr[:,:,0] = reds_01
    image_arr[:,:,1] = greens_01
    image_arr[:,:,2] = blues_01
    return image_arr


def X_pca_to_image_arr(new_image_X_pca, pca, size_to_crop_to):
    new_image_X_rgb = pca.inverse_transform(new_image_X_pca)
    image_reshape_shape = size_to_crop_to + (3,)  # RGB is last element
    new_image_arr = new_image_X_rgb.reshape(image_reshape_shape)
    new_image_arr = fix_clipping(new_image_arr)
    # print("new_image_arr", new_image_arr)
    return new_image_arr


def dump_sample_outputs_to_file(n_training_images, pca, X_pca, size_to_crop_to):
    output_dir = "ImageDatasets/PhonePhotosOutput"

    # basis images
    print("creating basis images")
    n_components = X_pca.shape[1]
    X_pca_max_abs_by_component = get_X_pca_max_abs_by_component(X_pca)
    output_datetime = datetime.utcnow()
    output_dt_str = output_datetime.strftime("%Y%m%d-%H%M%S")
    for ci in range(n_components):
        for sign in [-1, 1]:
            v = np.zeros((n_components,))
            v[ci] = sign * X_pca_max_abs_by_component[ci]  # tried making it just 1, but they all looked very similar due to being in the center of the vector space
            new_image_arr = X_pca_to_image_arr(v, pca, size_to_crop_to)
            output_fname = output_dt_str + "-basis{}{}.jpg".format(ci, "_pn"[sign])
            output_fp = os.path.join(output_dir, output_fname)

            imshow_image_only(new_image_arr, output_fp, save=True)
            plt.close()  # clear memory of plot

    # random new images
    for i in range(100):
        print("creating image {}".format(i))
        # style = random.choice(["average", "random_point"])
        style = "random_point"
        if style == "average":
            new_image_X_pca = get_average_of_images(4, n_training_images, X_pca)
        elif style == "random_point":
            new_image_X_pca = get_random_image_in_pca_space(n_components, X_pca)
        else:
            raise Exception("style {}".format(style))
    
        new_image_arr = X_pca_to_image_arr(new_image_X_pca, pca, size_to_crop_to)

        # plt.imshow(new_image_arr)
        output_fname = output_dt_str + ".jpg"
        output_fp = os.path.join(output_dir, output_fname)

        # for when it makes multiple images in the same second
        disambig_output_fp = output_fp
        disambig_int = 1
        while os.path.exists(disambig_output_fp):
            disambig_output_fp = output_fp.replace(".", "_{}.".format(disambig_int))
            disambig_int += 1

        imshow_image_only(new_image_arr, disambig_output_fp, save=True)
        # plt.show()
        plt.close()  # clear memory of plot


def imshow_image_only(image_arr, fp, save=False):
    # https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    fig = plt.figure(frameon=False)
    fig.set_size_inches(4,4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_arr, aspect='auto')
    fig.savefig(fp, dpi=100)



if __name__ == "__main__":
    n_training_images = 100
    size_to_crop_to = (200, 200)

    X = phonephotos.get_training_data(n_training_images, size_to_crop_to, max_val=1)

    print("fitting PCA")
    n_components = 30
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    # X_new = pca.inverse_transform(X_pca)  # this is how you take a point in the PCA's basis and then get the real values back

    print("creating output images")
    dump_sample_outputs_to_file(n_training_images, pca, X_pca, size_to_crop_to)
    print("done")
