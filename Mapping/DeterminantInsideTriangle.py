# from https://stackoverflow.com/a/13933736/7376935
# not sure of a proof of this, or where the OP got it, gonna test it

import numpy as np
import matplotlib.pyplot as plt
from UnitSpherePoint import UnitSpherePoint


def point_is_in_triangle(p, triangle_points):
    a = point_is_in_triangle_determinant_test(p, triangle_points)
    # b = point_is_outside_triangle_anticentroid_test(p, triangle_points)
    return a


def point_is_in_triangle_determinant_test(p, triangle_points):
    # want the det signs all to be the same, and also the same as those for the centroid (so we are not on the opposite side of the planet)
    centroid = get_centroid(triangle_points)
    centroid_det_signs = get_det_signs(centroid, triangle_points)
    p_det_signs = get_det_signs(p, triangle_points)
    cs0, cs1, cs2 = centroid_det_signs
    assert cs0 == cs1 == cs2, "centroid outside triangle"
    ps0, ps1, ps2 = p_det_signs
    return ps0 == ps1 == ps2 == cs0


def point_is_in_quadrilateral(ptest, p00, p01, p10, p11):
    # need the quadrilateral points in the correct order, i.e.,
    # p00 is the upper left of the image (min row, min column)
    # p01 is the upper right of the image (min row, max column)
    # p10 is the lower left of the image (max row, min column)
    # p11 is the lower right of the image (max row, max column)
    # we will pick two triangles that compose the quadrilateral
    # triangle 1: p00, p01, p10
    # triangle 2: p11, p01, p10
    tps1 = [p00, p01, p10]
    tps2 = [p11, p01, p10]
    in_t1 = point_is_in_triangle(ptest, tps1)
    in_t2 = point_is_in_triangle(ptest, tps2)
    return in_t1 or in_t2


def get_det_signs(p, triangle_points):
    t1, t2, t3 = triangle_points
    m1 = np.stack([p, t2, t3])  # OP wrote t1,t2,p; I believe this is a typo for p,t2,t3; based on his code below in the answer, it is indeed supposed to have p replacing t1 in the first matrix
    m2 = np.stack([t1, p, t3])
    m3 = np.stack([t1, t2, p])
    d1 = np.linalg.det(m1)
    d2 = np.linalg.det(m2)
    d3 = np.linalg.det(m3)
    s1 = np.sign(d1)
    s2 = np.sign(d2)
    s3 = np.sign(d3)
    return [s1, s2, s3]


def point_is_outside_triangle_anticentroid_test(p, triangle_points):
    centroid = get_centroid(triangle_points)
    anticentroid = -1 * centroid  # center of sphere is origin
    t1, t2, t3 = triangle_points
    tps1 = [anticentroid, t2, t3]
    tps2 = [t1, anticentroid, t3]
    tps3 = [t1, t2, anticentroid]
    p_in_triangle_1 = point_is_in_triangle_determinant_test(p, tps1)
    p_in_triangle_2 = point_is_in_triangle_determinant_test(p, tps2)
    p_in_triangle_3 = point_is_in_triangle_determinant_test(p, tps3)
    return p_in_triangle_1 or p_in_triangle_2 or p_in_triangle_3


def get_centroid(triangle_points):
    assert len(triangle_points) == 3
    centroid = 1/3 * (sum(triangle_points))
    assert centroid.shape == (3,)
    centroid /= np.linalg.norm(centroid)  # normalize
    return centroid


def get_triangle_points():
    return [UnitSpherePoint.get_random_unit_sphere_point().xyz(as_array=True) for i in range(3)]


def plot_sphere_mesh(fig, ax):
    # copied from https://stackoverflow.com/questions/51645694/how-to-plot-a-perfectly-smooth-sphere-in-python-using-matplotlib
    N=25
    stride=2
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    r = 0.6
    x *= r
    y *= r
    z *= r
    ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride, color="grey")


def test_centroid_inside_triangle():
    for i in range(1000):
        triangle_points = get_triangle_points()
        centroid = get_centroid(triangle_points)
        assert point_is_in_triangle(centroid, triangle_points)


def scatter_3d_test():
    triangle_points = get_triangle_points()
    test_points = [UnitSpherePoint.get_random_unit_sphere_point().xyz(as_array=True) for i in range(1000)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_sphere_mesh(fig, ax)
    cx, cy, cz = get_centroid(triangle_points)
    ax.scatter(cx, cy, cz, color="black")
    for p in test_points:
        x,y,z = p
        in_triangle = point_is_in_triangle(p, triangle_points)
        color = "yellow" if in_triangle else "green"
        ax.scatter(x, y, z, color=color)
    plt.show()


if __name__ == "__main__":
    test_centroid_inside_triangle()
    scatter_3d_test()
