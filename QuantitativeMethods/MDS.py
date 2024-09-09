import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


def mds_plot_2d_from_array(arr, ax=None, colors=None, marker_to_indices=None):
    mds_fit = MDS(n_components=2).fit_transform(arr)
    xs, ys = mds_fit.T
    if ax is None:
        fig, ax = plt.subplots()

    if marker_to_indices is None:
        ax.scatter(xs, ys, c=colors)
    else:
        for marker, indices in marker_to_indices.items():
            these_colors = [colors[i] for i in indices]
            ax.scatter(xs[indices], ys[indices], c=these_colors, marker=marker)

    ax.set_aspect('equal')
    ax.set_title("MDS")
    ax.set_xticks([])
    ax.set_yticks([])

    return xs, ys

