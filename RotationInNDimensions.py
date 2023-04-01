# thinking about "super-continua" where there is an uncountable number of dimensions and no preferred basis (all decompositions of the space into some set of dimensions are arbitrary)
# try visualizing rotation of a point in N-dimensional space
# and as N gets large, might help me visualize what rotation (change of basis? sort of) in infinite-dimensional space might look like

import numpy as np
import matplotlib.pyplot as plt
import math

from InteractivePlot import InteractivePlot


def get_rotation_matrix_span_uv(u, v, theta):
    # https://math.stackexchange.com/questions/197772/generalized-rotation-matrix-in-n-dimensional-space-around-n-2-unit-vector
    # returns matrix that rotates the span of orthonormal (orthogonal 1-length) vectors u and v by angle theta
    assert len(u.shape) == 1
    ndim, = u.shape
    assert v.shape == (ndim,)
    u = u.reshape((ndim, 1))
    v = v.reshape((ndim, 1))
    I = np.identity(ndim)
    A = I + np.sin(theta)*(v*u.T - u*v.T) + (np.cos(theta)-1)*(u*u.T + v*v.T)
    return A


if __name__ == "__main__":
    ndim = 800
    p = np.cumsum(np.random.uniform(-1, 1, (ndim,)))

    # smooth the curve of p out a bit
    sigma = 10
    window = 100
    kernel = np.array([1/(sigma * (2*np.pi)**0.5) * np.exp(-(x/sigma)**2) for x in range(-window, window+1)])
    kernel /= np.linalg.norm(kernel)
    p = np.convolve(p, kernel, mode="same")

    u = np.random.uniform(-1, 1, (ndim,))

    # get v such that u dot v = 0, choose first n-1 components of v
    v1 = np.random.uniform(-1, 1, (ndim-1,))
    dot_so_far = np.dot(u[:-1], v1)
    # then last element of u times last element of v has to cancel this dot out exactly
    # later if want to make this choice of v more fair, can choose the last index randomly rather than the rightmost one
    ulast = u[-1]
    vlast = -1 * dot_so_far / ulast
    v = np.array(list(v1) + [vlast])
    # now normalize u and v to magnitude 1
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    # check that u and v are orthonormal
    assert abs(np.linalg.norm(u) - 1) < 1e-9
    assert abs(np.linalg.norm(v) - 1) < 1e-9
    assert abs(np.dot(u, v) - 0) < 1e-9, f"u and v are not orthogonal:\n{u=}\n{v=}\ndot = {np.dot(u,v)}"

    thetas = np.arange(0, 2*np.pi, 2*np.pi/60)
    As = [get_rotation_matrix_span_uv(u, v, theta) for theta in thetas]
    p2s = [np.matmul(A, p) for A in As]
    max_y = max(p2.max() for p2 in p2s)
    min_y = min(p2.min() for p2 in p2s)
    a = 1/2 * (min_y + max_y)
    r = 1/2 * (max_y - min_y)
    c = 1.1
    y0 = a - c*r
    y1 = a + c*r
    with InteractivePlot() as plt:
        while True:
            for p2 in p2s:
                plt.plot(range(ndim), p2, c="r", alpha=0.75)
                plt.plot(range(ndim), p, c="b", alpha=0.5)
                plt.ylim((y0, y1))
                plt.draw()
                plt.gcf().clear()

