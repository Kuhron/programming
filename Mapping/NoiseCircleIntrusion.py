import math
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import KDTree

import NoiseMath as nm

import sys
sys.path.insert(0,'..')  # cause I can't be bothered to make packages for all these separate things
from PltContentOnly import imshow_content_only, scatter_content_only, add_opaque_background



def add_circle_of_points(center, radius, decay_exponent, n_points, existing_points):
    # map existing points to same angle from the center, with increased distance as though they are pushed out of the way by the new bubble
    r_hat = radius * 1.01
    get_increased_distance = lambda d: d + r_hat * (decay_exponent ** -d)  # verified that this satisfies d2>d1 => d2'>d1'
    distances = [np.linalg.norm(p-center) for p in existing_points]
    dxs = [p[0] - center[0] for p in existing_points]
    dys = [p[1] - center[1] for p in existing_points]
    thetas = [math.atan2(dy,dx) for dy,dx in zip(dys, dxs)]
    new_distances = [get_increased_distance(d) for d in distances]
    new_xs = [center[0] + d_prime * math.cos(theta) for d_prime, theta in zip(new_distances, thetas)]
    new_ys = [center[1] + d_prime * math.sin(theta) for d_prime, theta in zip(new_distances, thetas)]
    new_existing_points = [np.array([new_x, new_y]) for new_x, new_y in zip(new_xs, new_ys)]

    thetas_for_bubble_points = np.arange(0, 2*np.pi, 2*np.pi/n_points)
    bubble_xs = [center[0] + radius * math.cos(theta) for theta in thetas_for_bubble_points]
    bubble_ys = [center[1] + radius * math.sin(theta) for theta in thetas_for_bubble_points]
    bubble_points = [np.array([x,y]) for x,y in zip(bubble_xs, bubble_ys)]

    return new_existing_points + bubble_points


def filter_01_box(points):
    # keep only those points in the 01 box
    return [p for p in points if (0 <= p[0] <= 1) and (0 <= p[1] <= 1)]


def get_center_point():
    # return get_center_point_normal()
    return np.random.uniform(0, 1, (2,))


def get_center_point_normal():
    while True:
        x,y = np.random.normal(0.5, 0.25, (2,))
        if 0 < x < 1 and 0 < y < 1:
            return x,y


def get_point_set():
    points = []
    for i in range(120):
        center = get_center_point()
        radius = np.random.lognormal(np.log(0.05), np.log(1.5))
        n_points = 360 # np.random.lognormal(np.log(100), np.log(2))
        decay_exponent = np.exp(5) # max(1.01, np.random.lognormal(1, np.log(2)))
        points = add_circle_of_points(center, radius, decay_exponent, n_points, points)
        points = filter_01_box(points)
    return points


def get_distances_to_points(points, xy_mesh):
    kdtree = KDTree(np.array(points))
    X,Y = xy_mesh
    assert X.shape == Y.shape
    distances = np.zeros(X.shape)
    XY = np.stack([X, Y], axis=-1)  # kdtree.query wants last dimension to be the number of dims of a point (point-array-first format)
    ds, p_indexes = kdtree.query(XY)
    return ds


def create_scatter_points_image(points, out_fp):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    scatter_content_only(xs, ys, save_fp=out_fp, background_color="WHITE")


def create_distance_image(points, out_fp):
    xs = np.linspace(0, 1, 1000)
    ys = np.linspace(0, 1, 1000)
    xy_mesh = np.meshgrid(xs, ys)
    distances = get_distances_to_points(points, xy_mesh)
    bump_func = nm.get_bump_func()
    Z = bump_func(distances)
    imshow_content_only(Z, save_fp=out_fp)


def generate_images(n_images):
    for im_i in range(n_images):
        print(f"generating image {im_i}/{n_images}")
        points = get_point_set()
        now_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        scatter_out_fp = f"NoiseImages/CircleIntrusion/{now_str}-{im_i}-scatter.png"
        imshow_out_fp = f"NoiseImages/CircleIntrusion/{now_str}-{im_i}-imshow.png"
        # create_scatter_points_image(points, scatter_out_fp)
        create_distance_image(points, imshow_out_fp)


if __name__ == "__main__":
    generate_images(10)
