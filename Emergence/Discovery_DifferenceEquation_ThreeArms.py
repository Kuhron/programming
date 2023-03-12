# just trying to play with some differential equations and see what happens if I simulate the behavior
# want to get some cool shapes, weird curves, attractors
# shapes like the hair monsters on the wall, moss, cool natural patterns
# also want to evolve stuff phylogenetically (but continuously through division and further evolution) and see what clusters arise
# galaxy clusters, filaments


import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


def get_dv(v, dt):
    # can make it independent of dimensionality?
    dv = np.zeros(v.shape)
    a = np.zeros(v.shape)
    b = np.zeros(v.shape)
    c = np.zeros(v.shape)
    for i, x in enumerate(v):
        ai = i*x
        bi = 1/(i+1)
        ci = np.sqrt(abs(ai**2 - bi**2))
        a[i] = ai
        b[i] = bi
        c[i] = ci
    a2 = rot(a, 3)
    b2 = rot(b, 5)
    c2 = rot(c, 14)
    dv = a2 * dt + 3 * b2 * c2
    return dv


def change_vec(v, dt):
    dv = get_dv(v, dt)
    return v + dv


def rot(v, n):
    w = v.flatten()
    n = n % len(w)
    w = np.concatenate([w[n:], w[:n]])
    v = w.reshape(v.shape)
    return v


def filter_max_abs(vecs, max_abs):
    new = []
    for v in vecs:
        mag = np.linalg.norm(v)
        if mag <= max_abs:
            new.append(mag)
    return np.array(new)


def reduce_abs(vecs, max_abs):
    # apply restoring force back toward origin
    # map infinity in a direction to max_abs in that direction
    get_new_mag = lambda mag: max_abs * (1 - np.exp(-mag))
    mags = np.linalg.norm(vecs, axis=1).reshape(vecs.shape[0], 1)
    new_mags = get_new_mag(mags)
    r = new_mags / mags
    new_vecs = vecs * r
    return new_vecs



if __name__ == "__main__":
    n_points = 500
    point_ndim = 10
    vecs = np.random.uniform(-5, 5, (n_points, point_ndim))

    dt = 0.01
    total_dt = 10
    n_steps = int(round(total_dt / dt))
    max_abs = 100
    mds_ndim = 3
    for step in range(n_steps):
        vecs = change_vec(vecs, dt)
        # vecs = filter_max_abs(vecs, max_abs)
        vecs = reduce_abs(vecs, max_abs)

    mds = MDS(n_components=mds_ndim)
    X = mds.fit_transform(vecs)

    if mds_ndim == 2:
        xs, ys = X.T
        plt.scatter(xs, ys)
    elif mds_ndim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        xs, ys, zs = X.T
        ax.scatter(xs, ys, zs)
    else:
        raise ValueError(mds_ndim)

    plt.show()

