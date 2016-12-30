import sys
import random

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from images2gif import writeGif

# import matplotlib.pyplot as plt 
# import matplotlib.animation as animation
#import cv2

# def build_gif(imgs, show_gif=True, gif_filepath=None, title=""):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_axis_off()

#     ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

#     im_ani = animation.ArtistAnimation(fig, ims, interval=800, repeat_delay=0, blit=False)

#     if gif_filepath:
#         im_ani.save(gif_filepath, writer="imagemagick")
#         print("gif saved")

#     if show_gif:
#         plt.show()

#     return

def blend(filename1,filename2,alpha_percent):
    s1 = filename1.split(".")
    s2 = filename2.split(".")

    ext = s1[-1]
    if s2[-1] != ext:
        raise TypeError("Both images must be same file extension.")

    f1 = ".".join(s1[:-1])
    f2 = ".".join(s2[:-1])

    im_a = Image.open(DIRECTORY+"\\"+filename1)
    im_b = Image.open(DIRECTORY+"\\"+filename2)
    
    im_ab = Image.blend(im_a,im_b,alpha=float(alpha_percent)/100.0)
    full_filepath = DIRECTORY+"\\"+f1+"_"+f2+"_alpha"+str(alpha_percent)+"."+ext
    print("Saved to %s" % full_filepath)
    im_ab.save(full_filepath)

def blend_rgbs(a,b,alpha=0.5):
    scale = lambda x,y: int(alpha*x + (1-alpha)*y)
    scale_i = lambda i: scale(a[i],b[i])
    return tuple([scale_i(i) for i in range(3)])

def get_random_rgb_number():
    return random.randint(0,255)

def get_random_rgb():
    return tuple([get_random_rgb_number() for i in range(3)])

def get_contiguous_subsequences(lst):
    if len(lst) == 0:
        return []
    elif len(lst) == 1:
        return [lst]
    lst = sorted(lst)
    quantum = min([lst[i] - lst[i-1] for i in range(1,len(lst))])
    running_result = []

    running_result = [lst[0]]
    i = 1
    while i < len(lst) and lst[i] - lst[i-1] == quantum:
        running_result.append(lst[i])
        i += 1

    return [running_result] + get_contiguous_subsequences(lst[i:])

def get_stripe_colors(set):
    d = {}
    for subsequence in get_contiguous_subsequences(set):
        color = get_random_rgb()
        for element in subsequence:
            d[element] = color
    for i in set:
        if i not in d:
            raise Exception("bug in function get_contiguous_subsequences")
    return d

def get_random_bmp(dims,background="white"):
    im = Image.new("RGBA",dims,background)
    xs = [i for i in filter(lambda x: random.random() < coordinate_selection_probability, range(dims[0]))]
    ys = [i for i in filter(lambda x: random.random() < coordinate_selection_probability, range(dims[1]))]
    x_colors = get_stripe_colors(xs)
    y_colors = get_stripe_colors(ys)
    for x in xs:
        for y in ys:
            im.putpixel((x,y),blend_rgbs(x_colors[x],y_colors[y]))
    return im

def png_to_gif_ready(im):
    # background = Image.new("RGBA", im.size, "white")
    # background.paste(im, im)
    # im = background.convert("RGB").convert("P", palette=Image.ADAPTIVE)
    # im.save(DIRECTORY+"animation.gif") # just keep writing over what will end up as the final gif
    # im = Image.open(DIRECTORY+"animation.gif")
    return im


DIRECTORY = "C:\\Users\\Wesley\\Pictures\\Databending\\Brakhage\\"

dims = (400,400)
coordinate_selection_probability = 0.9
n_images = 5
background = "black"

images = [png_to_gif_ready(get_random_bmp(dims,background)) for i in range(n_images)]
for i in range(len(images)):
    images[i].save(DIRECTORY + "Brakhage" + str(i) + ".png")
#im = ImageSequence.Iterator(images)
#im.save(DIRECTORY + "Brakhage.gif", "gif")
#build_gif(images, show_gif=True, gif_filepath=DIRECTORY+"animation.gif")
#writeGif(DIRECTORY+"animation.gif",images,duration=1,dither=0)






