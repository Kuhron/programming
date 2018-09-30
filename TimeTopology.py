import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


W_INTERVAL = (0, 1)
in_interval = lambda x, iv: iv[0] <= x <= iv[1]
w_is_valid = lambda w0, w1: in_interval(w0, W_INTERVAL) and in_interval(w1, W_INTERVAL)


def zero_trajectory():
    while True:
        yield (0, 0)


def random_quadratic_trajectory(small=0.001, init=zero_trajectory()):
    a0 = random.uniform(-small, small)
    b0 = random.uniform(-small, small)
    a1 = random.uniform(-small, small)
    b1 = random.uniform(-small, small)
    t = 0
    while True:
        c = next(init)
        yield (c[0] + a0 * t + b0, c[1] + a1 * t + b1)
        t += 1


def tree():
    # timelines bifurcate sometimes and drift around, observer can jump to sufficiently nearby timelines with some probability

    # for now make points tuples of (w0, w1, g) where g is a generator giving the difference function of the point's trajectory in w-space
    # always have a point in the middle so things can bifurcate from it even if everything else walks off the edge
    initial_points = [
        (0.5, 0.5, zero_trajectory(), "k"),
        (0.75, 0.25, zero_trajectory(), "g"),
        (0.75, 0.75, zero_trajectory(), "b"),
        (0.25, 0.75, zero_trajectory(), "r"),
        (0.25, 0.25, zero_trajectory(), "y"),
    ]

    t = 0
    t_max = 20
    bifurcation_probability = 0.5
    segments_to_plot = []
    plt.subplot(1, 1, 1, projection='3d')

    # small = 0.01
    # move = lambda point: (point[0] + random.uniform(-small, small), point[1] + random.uniform(-small, small))
    def move(point, new_trajectory=False):
        trajectory = random_quadratic_trajectory(small=0.005, init=point[2]) if new_trajectory else point[2]
        diff = next(trajectory)
        return (point[0] + diff[0], point[1] + diff[1], trajectory, point[3])
    
    previous_points = []
    current_points = initial_points
    while t < t_max:
        previous_t = t
        t += 1
        print("tick {} of {}".format(t, t_max))
        previous_points = current_points
        current_points = []
        for point in previous_points:
            if random.random() < bifurcation_probability:
                # bifurcate
                new_points = [move(point), move(point, new_trajectory=True)]
            else:
                new_points = [move(point)]
            for new_point in new_points:
                if w_is_valid(new_point[0], new_point[1]):
                    current_points.append(new_point)
                    segments_to_plot.append([(point[0], point[1], previous_t, point[3]), (new_point[0], new_point[1], t, point[3])])

    n_segs = len(segments_to_plot)
    for i, seg in enumerate(segments_to_plot):
        # segment should be in format [(x0, y0, z0), (x1, y1, z1)]
        print("segment {} of {}".format(i, n_segs))
        p0, p1 = seg
        x0, y0, z0, c0 = p0
        x1, y1, z1, c1 = p1
        assert c0 == c1  # points on same tree should be same color

        # alpha should be inversely proportional to velocity so quickly-diverging timelines are not as salient
        alpha_0 = 0.5
        seg_length = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5  # don't use delta_z because it is always 1 and dominates the magnitude
        alpha = alpha_0 / (1 + 10*seg_length)
        plt.plot([x0, x1], [y0, y1], [z0, z1], color=c0, alpha=alpha)

    plt.show()


def solid_space():
    # all points are valid and the observer can move around in w continuously
    raise


if __name__ == "__main__":
    tree()
