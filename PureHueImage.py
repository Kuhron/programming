# convert image to HSV, then maximize saturation and value, leaving only "pure hue"
# because I was curious about the hues seen in an aerial image of twilight over the Ross Sea in Antarctica: https://lh3.googleusercontent.com/proxy/xiXOr8TtWYHKD9DVzRLBmnxY7OM-mE_735Q-8L1-0zVZRLayFDgO9RKLssq7Hj6XeTGMKcEr4HowFXizGfaAnzMWdcQQUTATeFlY-W2gaeAHAklbs4YumN6y-Sb1BoB6gQmWKA


from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import random

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

fp = filedialog.askopenfilename(initialdir="/home/wesley/Pictures/")
# try to kill Tk stuff so plt.show doesn't hang
root.quit()
root.destroy()

print("got fp: {}".format(fp))

with Image.open(fp) as im:
    arr = np.array(im.convert("RGB")) / 255  # convert to 0-1 so rgb_to_hsv doesn't create incorrect values (without the /255, it will make H and S in 0-1 but V in 0-255)

# def rgb2hsv(image):
#     return image.convert('HSV')
# im = rgb2hsv(im)

# look at sample of color number values, debug invalid range for V (0-255)
# for i in range(10):
#     r = random.randrange(arr2.shape[0])
#     c = random.randrange(arr2.shape[1])
#     print(arr2[r][c])

def to_pure_hue(rgb_arr):
    hsv_arr = matplotlib.colors.rgb_to_hsv(rgb_arr)
    hsv_arr[:,:,1] = 1.0
    hsv_arr[:,:,2] = 1.0
    res = matplotlib.colors.hsv_to_rgb(hsv_arr)
    return res

def to_max_saturation(rgb_arr):
    hsv_arr = matplotlib.colors.rgb_to_hsv(rgb_arr)
    hsv_arr[:,:,1] = 1.0
    res = matplotlib.colors.hsv_to_rgb(hsv_arr)
    return res

arr2 = to_max_saturation(arr)
arr3 = to_pure_hue(arr)

plt.subplot(1,2,1)
plt.imshow(arr)
plt.subplot(1,2,2)
plt.imshow(arr2)
# plt.subplot(2,2,3)
# plt.imshow(arr3)
plt.show()


# trying to get the damn thing to close without having to `kill %1` in shell
im.close()
sys.exit()
raise Exception("force exit")
# whatever, screw it, I'll just kill the process
