from PIL import Image
import numpy as np
import random
import os
import colorsys

from ImageDatasetLoadPhonePhotos import get_random_image_fps
from PerceptualHueScale import get_standard_hue_from_desired_01, get_desired_hue_from_standard_01


def rgb_to_hsv(r, g, b):
    # since vectorizing colorsys.rgb_to_hsv is slow
    # formulas: https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    assert r.shape == g.shape == b.shape
    assert (0 <= r).all() and (r <= 1).all()
    assert (0 <= g).all() and (g <= 1).all()
    assert (0 <= b).all() and (b <= 1).all()

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros(r.shape)
    h[delta == 0] = 0
    r_mask = (cmax == r) & (delta != 0)
    g_mask = (cmax == g) & (delta != 0)
    b_mask = (cmax == b) & (delta != 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        h[r_mask] = (60/360 * (((g-b)/delta) % 6))[r_mask]  # mask this one too so it can assign without throwing error that boolean indexing assignment needs to be 0D or 1D
        h[g_mask] = (60/360 * ((b-r)/delta + 2))[g_mask]
        h[b_mask] = (60/360 * ((r-g)/delta + 4))[b_mask]

        s = np.zeros(r.shape)
        s[cmax == 0] = 0
        s[cmax != 0] = (delta/cmax)[cmax != 0]

    v = cmax

    return h, s, v


def hsv_to_rgb(h, s, v):
    # since vectorizing colorsys.hsv_to_rgb is slow
    # formulas: https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    assert h.shape == s.shape == v.shape
    assert (0 <= h).all() and (h < 1).all()  # h strictly less than 360 deg
    assert (0 <= s).all() and (s <= 1).all()
    assert (0 <= v).all() and (v <= 1).all()
    c = v * s
    x = c * (1 - abs((h/(60/360)) % 2 - 1))
    m = v - c

    r = np.zeros(h.shape)
    g = np.zeros(h.shape)
    b = np.zeros(h.shape)

    h_0_60    = (0       <= h) & (h < 60/360)
    h_60_120  = (60/360  <= h) & (h < 120/360)
    h_120_180 = (120/360 <= h) & (h < 180/360)
    h_180_240 = (180/360 <= h) & (h < 240/360)
    h_240_300 = (240/360 <= h) & (h < 300/360)
    h_300_360 = (300/360 <= h) & (h < 1)

    r[h_0_60]    = c[h_0_60]
    r[h_60_120]  = x[h_60_120]
    r[h_120_180] = 0
    r[h_180_240] = 0
    r[h_240_300] = x[h_240_300]
    r[h_300_360] = c[h_300_360]

    g[h_0_60]    = x[h_0_60]
    g[h_60_120]  = c[h_60_120]
    g[h_120_180] = c[h_120_180]
    g[h_180_240] = x[h_180_240]
    g[h_240_300] = 0
    g[h_300_360] = 0

    b[h_0_60]    = 0
    b[h_60_120]  = 0
    b[h_120_180] = x[h_120_180]
    b[h_180_240] = c[h_180_240]
    b[h_240_300] = c[h_240_300]
    b[h_300_360] = x[h_300_360]

    r = r + m
    g = g + m
    b = b + m

    return r, g, b


def test_color_space_conversion(R,G,B):
    rh1 = np.vectorize(colorsys.rgb_to_hsv)
    hr1 = np.vectorize(colorsys.hsv_to_rgb)
    rh2 = rgb_to_hsv
    hr2 = hsv_to_rgb
    H1, S1, V1 = rh1(R,G,B)
    H2, S2, V2 = rh2(R,G,B)
    assert (abs(H1-H2) < 1e-9).all(), abs(H1-H2).max()
    assert (abs(S1-S2) < 1e-9).all(), abs(S1-S2).max()
    assert (abs(V1-V2) < 1e-9).all(), abs(V1-V2).max()
    H,S,V = H1, S1, V1
    R1, G1, B1 = hr1(H,S,V)
    R2, G2, B2 = hr2(H,S,V)

    assert (abs(R - R1) < 1e-9).all(), abs(R-R1).max()
    assert (abs(R - R2) < 1e-9).all(), abs(R-R2).max()
    assert (abs(G - G1) < 1e-9).all(), abs(G-G1).max()
    assert (abs(G - G2) < 1e-9).all(), abs(G-G2).max()
    assert (abs(B - B1) < 1e-9).all(), abs(B-B1).max()
    assert (abs(B - B2) < 1e-9).all(), abs(B-B2).max()
    print("color space conversion test passed\n")


def rotate_hue(im, rotation_deg):
    # rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    # hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    im = Image.open(fp)
    arr = np.array(im)
    # arr = arr[:100,:100,:]  # debug
    arr = arr / 256
    assert (0 <= arr).all() and (arr <= 1).all(), f"min: {arr.min()}, max: {arr.max()}"
    R,G,B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    H,S,V = rgb_to_hsv(R,G,B)
    # print("hue min:", H.min(), ", hue max:", H.max())
    rotation_01 = 1/360 * rotation_deg
    H_perceptual = get_desired_hue_from_standard_01(H)
    new_H_perceptual = ((H_perceptual + rotation_01) % 1)
    new_H = get_standard_hue_from_desired_01(new_H_perceptual)
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
    photo_dir = "/home/wesley/Desktop/IPhone Media/IPhone Media Temp Storage/"
    test_photos = [
        "2022-03-30/20220330_230210.jpg",  # done
        # "2022-11-12/20221112_221842.jpg",  # done
        # "2022-10-17/20221017_234754.jpg",  # done
        # "2022-10-15/20221015_210842.jpg",  # done
        # "2022-09-14/20220914_001143.jpg",  # done
    ]
    fps = [os.path.join(photo_dir, fp) for fp in test_photos]

    R = np.random.random((5000,))
    G = np.random.random((5000,))
    B = np.random.random((5000,))
    test_color_space_conversion(R,G,B)

    # n_images = 500
    # fps = get_random_image_fps(n_images)
    output_dir = "Images/HueRotation"
    for fp in fps:
        # rotation_deg = random.randrange(360)
        for rotation_deg in range(0, 360, 1):
            print(f"rotating hue by {rotation_deg:03d} deg: {fp}")
            im = Image.open(fp)
            im = rotate_hue(im, rotation_deg)
            fname = os.path.basename(fp)
            new_fname = f"{rotation_deg:03d}deg_" + fname
            new_fp = os.path.join(output_dir, new_fname)
            print(f"saved to {new_fp}")
            im.save(new_fp)

