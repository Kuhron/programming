# so I don't have to keep looking up this code

import matplotlib.pyplot as plt
from PIL import Image


def get_fig_ax(size_inches):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(*size_inches)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax


def imshow_content_only(arr, size_inches=(6,6), save_fp=None, **kwargs):
    # https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    fig, ax = get_fig_ax(size_inches)
    ax.imshow(arr, aspect="auto", **kwargs)
    if save_fp is not None:
        fig.savefig(save_fp)


def scatter_content_only(xs, ys, size_inches=(6,6), save_fp=None, background_color=None):
    fig, ax = get_fig_ax(size_inches)
    ax.scatter(xs, ys)
    if save_fp is not None:
        fig.savefig(save_fp)
        if background_color is not None:
            add_opaque_background(save_fp, background_color)


def add_opaque_background(image_fp, color):
    # https://stackoverflow.com/questions/50898034/how-replace-transparent-with-a-color-in-pillow
    im = Image.open(image_fp)
    image = Image.new("RGB", im.size, color)
    image.paste(im, (0, 0), im) 
    image.save(image_fp)

