import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


W_INTERVAL = (0, 1)
in_interval = lambda x, iv: iv[0] <= x <= iv[1]
w_is_valid = lambda w0, w1: in_interval(w0, W_INTERVAL) and in_interval(w1, W_INTERVAL)


def random_quadratic_trajectory():
    small = 0.01
    a0 = random.uniform(-small, small)
    b0 = random.uniform(-small, small)
    a1 = random.uniform(-small, small)
    b1 = random.uniform(-small, small)
    t = 0
    while True:
        yield (a0 * t + b0, a1 * t + b1)
        t += 1


def tree():
    # timelines bifurcate sometimes and drift around, observer can jump to sufficiently nearby timelines with some probability
    n_initial_points = 3

    # for now make points tuples of (w0, w1, g) where g is a generator giving the difference function of the point's trajectory in w-space
    initial_points = [(random.random(), random.random(), random_quadratic_trajectory()) for _ in range(n_initial_points)]

    t = 0
    t_max = 50
    bifurcation_probability = 0.1
    segments_to_plot = []
    plt.subplot(1, 1, 1, projection='3d')

    # small = 0.01
    # move = lambda point: (point[0] + random.uniform(-small, small), point[1] + random.uniform(-small, small))
    def move(point, new_trajectory=False):
        trajectory = random_quadratic_trajectory() if new_trajectory else point[2]
        diff = next(trajectory)
        return (point[0] + diff[0], point[1] + diff[1], trajectory)
    
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
                new_points = [move(point, new_trajectory=True), move(point, new_trajectory=True)]
            else:
                new_points = [move(point)]
            for new_point in new_points:
                if w_is_valid(new_point[0], new_point[1]):
                    current_points.append(new_point)
                    segments_to_plot.append([(point[0], point[1], previous_t,), (new_point[0], new_point[1], t,)])

    n_segs = len(segments_to_plot)
    for i, seg in enumerate(segments_to_plot):
        # segment should be in format [(x0, y0, z0), (x1, y1, z1)]
        print("segment {} of {}".format(i, n_segs))
        p0, p1 = seg
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        plt.plot([x0, x1], [y0, y1], [z0, z1], color="r", alpha=0.4)

    plt.show()


def solid_space():
    # all points are valid and the observer can move around in w continuously
    raise


if __name__ == "__main__":
    tree()
