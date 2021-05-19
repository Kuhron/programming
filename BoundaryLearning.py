# idea: create algorithm that learns a boundary of a set of points
# regardless of if points of other classes are also inside that boundary
# just care about the shape of the set under consideration
# use "control points" like in font making programs


import random
import math
import numpy as np
import matplotlib.pyplot as plt


class Boundary:
    def __init__(self, control_points):
        self.control_points = control_points  # must be in order of traversal

    def add_control_point(self, p, index):
        self.control_points = self.control_points[:index] + [p] + self.control_points[index:]

    def move_control_point(self, p, index):
        self.control_points[index] = p

    def contains(self, p):
        # is p inside the boundary
        raise NotImplementedError

    @staticmethod
    def initialize_from_samples(samples):
        # pick a random triangle in the samples
        starting_control_points = random.sample(samples, 3)
        return Boundary(starting_control_points)

    def plot(self, color):
        xys = self.control_points + [self.control_points[0]]  # wrap around
        xs = [p[0] for p in xys]
        ys = [p[1] for p in xys]
        plt.plot(xs, ys, c=color)


def get_points():
    colors = ["red"] #, "yellow", "blue"]  # one for each class
    n_classes = len(colors)
    center_xys = np.random.uniform(0, 1, (n_classes, 2))
    points = {}
    for color, center in zip(colors, center_xys):
        n_points_in_class = random.randint(6, 12)
        dxys = np.random.normal(0, 0.2, (n_points_in_class, 2))
        xys = center + dxys
        points[color] = list(xys)
    return points


def get_boundaries(points):
    boundaries = {}
    for color, xys in points.items():
        b = Boundary.initialize_from_samples(xys)
        boundaries[color] = b
    return boundaries


def plot_points(points):
    for color, xys in points.items():
        xs = [p[0] for p in xys]
        ys = [p[1] for p in xys]
        plt.scatter(xs, ys, c=color)


def plot_boundaries(boundaries):
    for color, boundary in boundaries.items():
        boundary.plot(color=color)


if __name__ == "__main__":
    points = get_points()
    boundaries = get_boundaries(points)

    plot_points(points)
    plot_boundaries(boundaries)
    plt.show()
