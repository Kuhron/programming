import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_plot_2d_from_array(arr, ax=None, colors=None, marker_to_indices=None):
    pca_fit = PCA(n_components=2).fit_transform(arr)
    xs, ys = pca_fit.T
    if ax is None:
        fig, ax = plt.subplots()

    # should probably make this part a function to avoid repetition, but annoying with 3d vs 2d (could just do 2 separate functions)
    if marker_to_indices is None:
        ax.scatter(xs, ys, c=colors)
    else:
        for marker, indices in marker_to_indices.items():
            these_colors = [colors[i] for i in indices]
            ax.scatter(xs[indices], ys[indices], c=these_colors, marker=marker)

    ax.set_aspect('equal')
    ax.set_title("PCA")
    ax.set_xticks([])
    ax.set_yticks([])


