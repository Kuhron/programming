import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import SampleDatasets as sam




if __name__ == "__main__":
    # X, n_clusters = sam.get_double_helix_dataset(500, radius=1, wavelength=np.pi/4, theta_offset=np.pi), 8
    # X, n_clusters = sam.get_cube_frame_dataset(500), 8
    # X, n_clusters = sam.get_sphere_boundary_dataset(500), 8
    # X, n_clusters = sam.get_multilobe_dataset(500, n_lobes=8)
    # X, n_clusters = sam.get_cube_corner_multilobe_dataset(500)
    # X, n_clusters = sam.get_multilobe_dataset_with_one_outside_plane(500)

    # or to load pre-made dataset
    dataset_dir = "/home/kuhron/programming/QuantitativeMethods/TestDatasets/"
    fnames = [
        "MdsPcaSimilar.txt",
        "CubeCornerLobes.txt",
        "MdsAvoidingOverlap.txt",
        "MdsAvoidingOverlap2.txt",
        "HelixRings.txt",
        "CubeFrame.txt",
    ]
    if "X" not in locals():  # hack to see if we've defined a dataset yet
        for i, fname in enumerate(fnames):
            print(f"{i+1}. {fname.replace('.txt','')}")
        choice = input("Choose a dataset by number: ")
        choice = int(choice) - 1
        fname = fnames[choice]
        print(f"you chose {fname}")
        fp = os.path.join(dataset_dir, fname)
        X, n_clusters = np.loadtxt(fp), 8

    do_clustering = n_clusters is not None
    if do_clustering:
        kmeans = KMeans(n_clusters=n_clusters).fit(X)
        cmap = plt.get_cmap("jet")
        # evenly space the colors through saturated hue space
        a = np.arange(0, 1, 1/n_clusters)
        a += random.random()
        a %= 1
        random.shuffle(a)  # in case adjacent cluster numbers are systematically next to each other in the space
        label_to_color = {k: cmap(a[k]) for k in range(n_clusters)}
        colors = [label_to_color[k] for k in kmeans.labels_]
    else:
        colors = None

    fig, axd = plt.subplot_mosaic([
            ["left", "upper right",],
            ["left", "lower right",],
        ], layout="constrained")
    axd["left"].set_xticks([])  # so they're not sticking out from underneath the 3d plot that I'm putting on top of it
    axd["left"].set_yticks([])
    axd["left"] = plt.subplot(1, 2, 1, projection='3d')

    sam.mds_plot_2d_from_array(X, ax=axd["upper right"], colors=colors)
    sam.pca_plot_2d_from_array(X, ax=axd["lower right"], colors=colors)
    sam.scatterplot_3d_from_array(X, ax=axd["left"], colors=colors)
    plt.show()

    if input("\nsave dataset? enter anything to save or just press enter otherwise") != "":
        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fp = f"/home/kuhron/programming/QuantitativeMethods/TestDatasets/{now}.txt"
        np.savetxt(fp, X)
        print(f"saved to {fp}")
    else:
        print("not saved")

