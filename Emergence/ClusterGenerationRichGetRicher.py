import random
import numpy as np
import matplotlib.pyplot as plt


def get_points_normal_distance_random_walk(n):
    # not fractal looking enough, too homogeneous
    points = []
    current = None
    for i in range(n):
        if current is None:
            current = np.random.random((2,))
        else:
            dx = np.random.normal(0,0.1,(2,))
            current += dx
            current %= 1
        points.append(list(current))
    points = np.array(points)
    assert points.shape == (n, 2)
    return points


def get_points_positive_reinforcement(n):
    # more likely to put a point close to existing points
    points = []
    for i in range(n):
        if len(points) == 0 or random.random() < 0.01:
            # choose new random point
            p = np.random.random((2,))
            points.append(list(p))
        else:
            # choose an existing point to deviate from
            # want rich-get-richer effect, such as find k closest neighbors and base the size of the random walk step on how crowded the neighborhood is (stay closer to them if they are closeby)
            # don't want to run expensive nearest-neighbor algorithm so just get some easy proxy for how crowded the area is
            ref_p = random.choice(points)
            if len(points) == 1:
                # no other points to look at
                stdev = 0.1
            else:
                dxs = sorted([abs(p[0]-ref_p[0]) for p in points if p != ref_p])
                dys = sorted([abs(p[0]-ref_p[0]) for p in points if p != ref_p])
                k = 5
                dxs = dxs[:k]
                dys = dys[:k]
                stdev = np.mean(dxs + dys)  # should concat lists
            dx, dy = np.random.normal(0, stdev, (2,))
            p = [(ref_p[0] + dx) % 1, (ref_p[1] + dy) % 1]
            points.append(p)
    points = np.array(points)
    assert points.shape == (n, 2)
    return points


if __name__ == "__main__":
    n = 5000

    get_reference_points = get_points_positive_reinforcement

    points = get_reference_points(n)
    xs = points[:,0]
    ys = points[:,1]
    plt.scatter(xs, ys)
    plt.gca().set_aspect("equal")
    plt.show()


