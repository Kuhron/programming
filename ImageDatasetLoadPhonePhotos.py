import os
from glob import glob
import random
import numpy as np
from PIL import Image


PHOTOS_DIR = "/home/wesley/Desktop/IPhone Media/IPhone Media Temp Storage"


def get_random_image_fps(n):
    # regardless of size
    image_files = glob(PHOTOS_DIR + "/**/*.jpg")
    return random.sample(image_files, n)


def initialize_size_report():
    image_files = glob(PHOTOS_DIR + "/**/*.jpg")
    print(len(image_files))
    sizes = {}
    size_record = ""
    for fp_i, fp in enumerate(image_files):
        if fp_i % 100 == 0:
            print("{}/{}".format(fp_i, len(image_files)))
        with Image.open(fp) as im:
            size = im.size
        if size not in sizes:
            sizes[size] = 0
        sizes[size] += 1
        size_record += "{}\t{}\n".format(fp, size)

    with open("ImageDatasets/PhonePhotos.txt", "w") as f:
        f.write(size_record)

    for k,v in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True):
        print("size {} occurred {} times".format(k,v))


def get_fps_to_use(desired_size):
    list_fp = "ImageDatasets/PhonePhotos.txt"
    with open(list_fp) as f:
        lines = f.readlines()
    ls = [l.strip().split("\t") for l in lines]
    res = []
    for l in ls:
        assert len(l) == 2, l
        size_str = l[1]
        assert size_str[0] == "("
        assert size_str[-1] == ")"
        size_str = size_str[1:-1]
        x_str, y_str = size_str.split(", ")
        x = int(x_str)
        y = int(y_str)
        if (x,y) == desired_size:
            fp = l[0]
            res.append(fp)
    return res


def get_image_arrays(fps, size_to_crop_to, max_val):
    assert max_val in [1, 255], max_val

    input("loading images. watch top memory usage. press enter to continue")
    arrays = []
    for fp in fps:
        with Image.open(fp) as im:
            smaller_im = im.resize(size_to_crop_to, resample=Image.BILINEAR)  # resampling methods: https://www.geeksforgeeks.org/python-pil-image-resize-method/
            arr = np.array(smaller_im)

        # print("sleeping briefly so top memory usage can update")
        # time.sleep(0.01)
        flat_arr = arr.reshape((arr.size,))  # make it just a 1D line because sklearn can't deal with high-rank arrays

        if max_val == 1:
            # rescale it to interval 0-1 instead of 0-255 so the pca will learn in that space, and then matplotlib won't incorrectly use the fact that the new image is float datatype to think that it should clip e.g. 254.1 to 1 (max of the float range 0-1 for matplotlib's interpretation of RGB float values)
            flat_arr = flat_arr/255  # if use /= will get error, see: https://github.com/numpy/numpy/issues/10565
        elif max_val == 255:
            pass
        else:
            raise ValueError(max_val)

        arrays.append(flat_arr)
    return np.array(arrays)


def get_training_data(n_training_images, size_to_crop_to, max_val):
    most_common_size = (4032, 3024)  # 8092 of my photos as of 2021-03-21 have this size
    fps_to_use = get_fps_to_use(most_common_size)
    fps_to_use = random.sample(fps_to_use, n_training_images)  # start with small size first

    size_to_crop_to = (200,200)  # giant image vectors use up tons of memory when PCA tries to fit
    X = get_image_arrays(fps_to_use, size_to_crop_to, max_val)
    return X


if __name__ == "__main__":
    # initialize_size_report()  # already done, go read the file instead: ImageDatasets/PhonePhotos.txt
    pass
