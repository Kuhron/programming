# generate some sample shapes of data points for testing methods on them

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA


def get_rotation_about_x_axis(theta=None):
    if theta is None:
        theta = np.random.uniform(0, 2*np.pi)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])
    return M


def get_rotation_about_y_axis(theta=None):
    if theta is None:
        theta = np.random.uniform(0, 2*np.pi)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])
    return M


def get_rotation_about_z_axis(theta=None):
    if theta is None:
        theta = np.random.uniform(0, 2*np.pi)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])
    return M


def get_3d_rotation_matrix():
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    yaw = get_rotation_about_z_axis()
    pitch = get_rotation_about_y_axis()
    roll = get_rotation_about_x_axis()
    M = yaw @ pitch @ roll
    return M


def get_points_on_helix(n_points, radius=1, wavelength=2*np.pi, min_theta=-2*np.pi, max_theta=2*np.pi):
    # assume axis is the z axis, can apply rotation to move it later
    def get_xyz_from_theta(theta):
        x = np.cos(theta)
        y = np.sin(theta)
        z_traversed_per_radian = wavelength / (2*np.pi)
        z = theta * z_traversed_per_radian
        return [x, y, z]  # keep as list and then we'll construct np.array right before returning, so we don't have array as element of array
    thetas = np.random.uniform(min_theta, max_theta, (n_points,))
    arr = []
    for theta in thetas:
        arr.append(get_xyz_from_theta(theta))
    return np.array(arr)


def get_double_helix_dataset(n_points, radius=1, wavelength=2*np.pi, min_theta=-2*np.pi, max_theta=2*np.pi, theta_offset=np.pi/2):
    np1 = n_points // 2
    np2 = n_points - np1

    h1 = get_points_on_helix(np1, radius=radius, wavelength=wavelength, min_theta=min_theta, max_theta=max_theta)
    h2 = get_points_on_helix(np2, radius=radius, wavelength=wavelength, min_theta=min_theta, max_theta=max_theta)
    # rotate h2 90 deg so it's a double helix shape
    r90 = get_rotation_about_z_axis(theta=theta_offset)
    h2 = (r90 @ h2.T).T
    h = np.concatenate([h1, h2], axis=0)

    M = get_3d_rotation_matrix()
    h = (M @ h.T).T  # apply rotation to points as column vectors
    return h


def get_cube_frame_dataset(n_points):
    # 12 edges, choose randomly each time
    # can de-loop this later if it's inefficient, but MDS and PCA should be way more intensive than this so it's probably fine
    arr = []
    one = lambda: random.choice([-1, 1])
    r = lambda: random.uniform(-1, 1)
    for i in range(n_points):
        if random.random() < 1/3:
            # choose (x,y) and vary z
            p = [one(), one(), r()]
        elif random.random() < 1/2:
            # choose (x,z) and vary y
            p = [one(), r(), one()]
        else:
            # choose (y,z) and vary x
            p = [r(), one(), one()]
        arr.append(p)
    return np.array(arr)


def get_sphere_boundary_dataset(n_points):
    arr = np.zeros((n_points, 3))
    # I'm sure there's a better way to do this but whatever, just rotate a bunch of unit vectors
    # can also use this to check for uniformity of the rotation matrix distribution (impressionistically)
    for i in range(n_points):
        p = np.array([1, 0, 0])
        M = get_3d_rotation_matrix()
        p = (M @ p.T).T
        arr[i,:] = p
    return arr


def get_centers_and_stdevs_of_lobes(n_lobes=None):
    # start with something like a 3d random walk to get the lobe centers
    # assign each lobe a different stdev
    centers = []
    stdevs = []
    if n_lobes is None:
        n_lobes = random.randrange(6, 13)
    c01 = lambda: random.choice([0, 1])
    for i in range(n_lobes):
        if i == 0:
            center = [0, 0, 0]
        else:
            base_center = random.choice(centers)
            while True:
                m = random.choice([1]*1 + [2]*0 + [3]*0)  # occasionally jump farther away
                allow_diagonal_jumps = True
                if allow_diagonal_jumps:
                    direction = [m*c01(), m*c01(), m*c01()]
                else:
                    direction = [1, 0, 0]
                    random.shuffle(direction)  # hack to choose x/y/z
                new_center = [base_center[0]+direction[0], base_center[1]+direction[1], base_center[2]+direction[2]]
                if new_center not in centers:
                    center = new_center
                break
        centers.append(center)
        stdev = np.random.uniform(0.1, 0.35)
        stdevs.append(stdev)
    return centers, stdevs


def get_cube_corner_multilobe_dataset(n_points):
    centers = [
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ]
    stdevs = [0.25] * 8
    return get_multilobe_dataset(n_points, centers=centers, stdevs=stdevs, n_lobes=8)


def get_multilobe_dataset_with_one_outside_plane(n_points):
    # all centers are in the z=0 plane except one
    # make sure spread in x and y directions is significantly more than the gap from the plane to the outlier
    mx = 2
    my = 2
    centers = [
        [0, 0, 0],
        [0, my, 0],
        [mx, 0, 0],
        [mx, my, 0],
        [mx, 2*my, 0],
        [2*mx, my, 0],
        [2*mx, 2*my, 0],
        [0, 0, 2],  # outlier sitting on top of the plane, also near the edge
    ]
    stdevs = [0.25] * 8
    # actually this doesn't seem to work when the outlier is over the center of the plane! ugh! MDS was supposed to be able to handle this
    return get_multilobe_dataset(n_points, centers=centers, stdevs=stdevs, n_lobes=8)


def get_multilobe_dataset(n_points, centers=None, stdevs=None, n_lobes=None):
    # trying to make an interesting shape with multimodal structure connected into a blob made of multiple lobes
    # each lobe is a spherical distribution around an integer lattice point

    if centers is None:
        assert stdevs is None
        centers, stdevs = get_centers_and_stdevs_of_lobes(n_lobes)
    else:
        assert len(centers) == len(stdevs) == n_lobes

    # now generate the points around these lobes
    arr = []
    for i in range(n_points):
        c_i = random.randrange(n_lobes)
        center = centers[c_i]
        stdev = stdevs[c_i]
        x,y,z = np.random.normal(0, stdev, (3,))
        p = [center[0]+x, center[1]+y, center[2]+z]
        arr.append(p)
    return np.array(arr), n_lobes


def scatterplot_3d_from_array(arr, ax=None, colors=None, marker_to_indices=None):
    xs, ys, zs = arr.T
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    if marker_to_indices is None:
        ax.scatter(xs, ys, zs, c=colors)
    else:
        for marker, indices in marker_to_indices.items():
            these_colors = [colors[i] for i in indices]
            ax.scatter(xs[indices], ys[indices], zs[indices], c=these_colors, marker=marker)

    ax.set_aspect('equal')
    ax.set_title("Actual dataset in 3D")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


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


