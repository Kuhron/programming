# the Fire.py program output great images but the fire intensity scale jumps around too much in the video
# so I'm gonna remove them from the images but keep the images otherwise the same


from PIL import Image
import os
import numpy as np


def replace_mask_with_white(fname, dir_name, positions_to_fix):
    fp = os.path.join(dir_name, fname)
    white = (255, 255, 255, 255)
    with Image.open(fp) as im:
        arr = np.array(im)
        for r, c in positions_to_fix:
            arr[r, c] = white
        new_fname = "new_" + fname
        new_fp = os.path.join(dir_name, new_fname)
        new_im = Image.fromarray(arr)
        new_im.save(new_fp)


mask_fp = "Images/Fire/firescalemask.png"
with Image.open(mask_fp) as mask_im:
    w = mask_im.width
    h = mask_im.height
    mask_data = mask_im.getdata()  # rgba, flattened in row-major
    white = (255, 255, 255, 255)
    green = (38, 127, 0, 255)
    positions_to_fix = []
    for i, tup in enumerate(mask_data):
        row, col = divmod(i, w)
        if tup != white:
            # print(f"pixel #{i}, ({row}, {col}), is not white: {tup}")
            positions_to_fix.append((row, col))
            if tup != green:
                input("^^^ go fix this one")  # stupid Pinta anti-aliasing on pencil tool


parent_dir = "Images/Fire/"
subdirs = [x for x in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, x)) and x.startswith("InteractivePlot")]

for this_dir in subdirs:
    dir_name = os.path.join(parent_dir, this_dir)
    for fname in os.listdir(dir_name):
        if not fname.endswith(".png"):
            continue
        print(fname)
        replace_mask_with_white(fname, dir_name, positions_to_fix)

