# generate some sample shapes of data points for testing methods on them

import math
import random
import numpy as np
import matplotlib.pyplot as plt


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


def get_2d_rotation_matrix(theta=None):
    if theta is None:
        theta = np.random.uniform(0, 2*np.pi)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([
        [ c, s],
        [-s, c],
    ])
    return M


def get_linear_transformation(*, n_dims, determinant_one=False):
    while True:
        M = np.random.normal(0, 1, (n_dims, n_dims))
        det = np.linalg.det(M)
        if det == 0:
            continue
        if determinant_one:
            # let it be -1 too
            M /= abs(det)
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


def get_centers_and_stdevs_of_lobes(*, n_dims, n_lobes=None):
    # start with something like a random walk to get the lobe centers
    # assign each lobe a different stdev
    centers = []
    stdevs = []
    if n_lobes is None:
        n_lobes = random.randrange(6, 13)
    c01 = lambda: random.choice([0, 1])
    for i in range(n_lobes):
        if i == 0:
            center = [0] * n_dims
        else:
            base_center = random.choice(centers)
            while True:
                m = random.choice([1]*1 + [2]*0 + [3]*0)  # occasionally jump farther away
                allow_diagonal_jumps = True
                if allow_diagonal_jumps:
                    direction = [m*c01() for d_i in range(n_dims)]
                else:
                    direction = [0] * n_dims
                    direction[random.randrange(n_dims)] == 1  # choose a random dimension
                new_center = [base_center[d_i]+direction[d_i] for d_i in range(n_dims)]
                if new_center in centers:
                    base_center = new_center
                else:
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
    return get_multilobe_dataset(n_points, centers=centers, stdevs=stdevs)


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
    return get_multilobe_dataset(n_points, centers=centers, stdevs=stdevs)


def get_multilobe_dataset(*, n_points, n_dims, centers=None, stdevs=None, n_lobes=None):
    # trying to make an interesting shape with multimodal structure connected into a blob made of multiple lobes
    # each lobe is a spherical distribution around an integer lattice point

    if n_dims not in [2, 3]:
        raise ValueError(f"unsupported number of dimensions: {n_dims} (should be 2 or 3)")

    if centers is not None and n_lobes is not None:
        assert len(centers) == n_lobes, f"number of centers ({len(centers)} and number of lobes ({n_lobes}) do not match"
    elif centers is None:
        assert stdevs is None
        centers, stdevs = get_centers_and_stdevs_of_lobes(n_lobes=n_lobes, n_dims=n_dims)
    else:
        assert len(centers) == len(stdevs) == n_lobes

    # now generate the points around these lobes
    arr = []
    for i in range(n_points):
        c_i = random.randrange(n_lobes)
        center = centers[c_i]
        stdev = stdevs[c_i]
        if n_dims == 2:
            x,y = np.random.normal(0, stdev, (2,))
            p = [center[0]+x, center[1]+y]
        elif n_dims == 3:
            x,y,z = np.random.normal(0, stdev, (3,))
            p = [center[0]+x, center[1]+y, center[2]+z]
        arr.append(p)
    return np.array(arr), n_lobes


def scatterplot_2d_from_array(arr, ax=None, colors=None, marker_to_indices=None, title=None):
    xs, ys = arr.T
    if ax is None:
        fig, ax = plt.subplots()

    if marker_to_indices is None:
        ax.scatter(xs, ys, c=colors)
    else:
         for marker, indices in marker_to_indices.items():
            these_colors = [colors[i] for i in indices]
            ax.scatter(xs[indices], ys[indices], c=these_colors, marker=marker)

    ax.set_aspect('equal')
    if title is None:
        title = "Actual dataset in 2D"
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def scatterplot_3d_from_array(arr, ax=None, colors=None, marker_to_indices=None, title=None):
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
    if title is None:
        title = "Actual dataset in 3D"
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def scatterplot_from_array(arr, ax=None, colors=None, marker_to_indices=None, title=None):
    n_points, n_dims = arr.shape
    if n_dims == 2:
        scatterplot_2d_from_array(arr, ax=ax, colors=colors, marker_to_indices=marker_to_indices, title=title)
    elif n_dims == 3:
        scatterplot_3d_from_array(arr, ax=ax, colors=colors, marker_to_indices=marker_to_indices, title=title)
    else:
        raise ValueError(f"data has {n_dims} dimensions; cannot scatter")



if __name__ == "__main__":
    n_dims = 2
    n_points = 1000
    n_lobes = 15
    arr, n_lobes_got = get_multilobe_dataset(n_points=n_points, n_dims=n_dims, centers=None, stdevs=None, n_lobes=n_lobes)
    scatterplot_from_array(arr, ax=None, colors=None, marker_to_indices=None)

    def savefig(title):
        plt.gcf().set_size_inches((12, 12))
        plt.gcf().tight_layout()
        plt.savefig(title)

    savefig("dataset.png")

    for i in range(5):
        M = get_linear_transformation(n_dims=n_dims)
        a = (M @ arr.T).T
        scatterplot_from_array(a, ax=None, colors=None, marker_to_indices=None, title="dataset linearly transformed")
        savefig(f"dataset_warped_{i}.png")

    import DimensionCorrelation as dc

    points = arr.T
    M = dc.find_cost_minimizing_transformation_brute_force(points)
    points = M @ points
    points = dc.normalize_variances(points)

    print(f"final correlation cost = {dc.get_correlation_cost(points)}")
    # see how rotation affects this; hypothesis: if correlations are exactly zero, then rotation will not change this (I sure hope that's true! if not then this approach has a problem where changing basis to another equally good orthogonal one can make the dataset correlated again)
    for theta_deg in range(0, 360, 15):
        theta = theta_deg * np.pi/180
        R = get_2d_rotation_matrix(theta)
        p2 = R @ points
        cost = dc.get_correlation_cost(p2)
        print(f"{theta_deg} deg: {cost = :.12f}")

    a = points.T
    scatterplot_from_array(a, ax=None, colors=None, marker_to_indices=None, title="dataset with minimized correlation between dimensions")
    savefig(f"dataset_pried_apart.png")

