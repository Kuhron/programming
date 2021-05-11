# so I don't have to keep looking up this code

import matplotlib.pyplot as plt


def imshow_content_only(arr, size_inches=(6,6), save_fp=None):
    # https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    fig = plt.figure(frameon=False)
    fig.set_size_inches(*size_inches)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, aspect="auto")

    if save_fp is not None:
        fig.savefig(save_fp)
