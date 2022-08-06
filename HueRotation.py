from PIL import Image
import numpy as np
import random
import os
import colorsys

from ImageDatasetLoadPhonePhotos import get_random_image_fps


def rotate_hue(im, rotation_deg):
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    im = Image.open(fp)
    arr = np.array(im)
    # arr = arr[:100,:100,:]  # debug
    arr = arr / 256
    assert (0 <= arr).all() and (arr <= 1).all(), f"min: {arr.min()}, max: {arr.max()}"
    R,G,B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    H,S,V = rgb_to_hsv(R,G,B)
    # print("hue min:", H.min(), ", hue max:", H.max())
    rotation_01 = 1/360 * rotation_deg
    new_H = ((H + rotation_01) % 1)
    new_R, new_G, new_B = hsv_to_rgb(new_H, S, V)
    new_arr = np.zeros(arr.shape)
    new_arr[:, :, 0] = new_R
    new_arr[:, :, 1] = new_G
    new_arr[:, :, 2] = new_B
    new_arr = (256 * new_arr).astype(np.uint8)
    # print(new_arr)
    # print(new_arr.shape)
    new_im = Image.fromarray(new_arr)
    return new_im


if __name__ == "__main__":
    n_images = 500
    fps = get_random_image_fps(n_images)
    output_dir = "Images/HueRotation"
    for fp in fps:
        rotation_deg = random.randrange(360)
        print(f"rotating hue by {rotation_deg} deg: {fp}")
        im = Image.open(fp)
        im = rotate_hue(im, rotation_deg)
        fname = os.path.basename(fp)
        new_fname = f"{rotation_deg}deg_" + fname
        new_fp = os.path.join(output_dir, new_fname)
        print(f"saved to {new_fp}")
        im.save(new_fp)
