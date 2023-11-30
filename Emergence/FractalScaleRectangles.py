import random
import numpy as np
import matplotlib.pyplot as plt

from ClusterGenerationRichGetRicher import get_points_positive_reinforcement


def get_line_directions(n):
    return np.random.choice([0,1], (n,))


def get_line_segment_endpoints(reference_points, verticalities):
    # draw horiz or vertical line through this reference point that goes from the point in both directions until it hits another line or the edge
    # if it hits an endpoint of an existing line, pass through it

    existing_horizontal_line_endpoints = [((0,0),(1,0)), ((0,1),(1,1))]
    existing_vertical_line_endpoints = [((0,0),(0,1)), ((1,0),(1,1))]

    for p, v in zip(reference_points, verticalities):
        # get the perpendicular lines that this one would cross if it extends forever
        # then find which one it will hit first on each side
        x,y = p
        if v == 0:
            # horizontal line
            left_crossers = []
            right_crossers = []
            rightmost_left_x = None
            leftmost_right_x = None
            for p0, p1 in existing_vertical_line_endpoints:
                # y is constant on the new line so we want to check their y ranges
                x0, y0 = p0
                x1, y1 = p1
                assert x0 == x1
                if y0 < y < y1:
                    # it's a crosser
                    if x0 < x:
                        left_crossers.append((p0, p1))
                        rightmost_left_x = max(rightmost_left_x, x0) if rightmost_left_x is not None else x0
                    elif x0 > x:
                        right_crossers.append((p0, p1))
                        leftmost_right_x = min(leftmost_right_x, x0) if leftmost_right_x is not None else x0
                    else:
                        # it can pass through this line
                        pass
            assert rightmost_left_x is not None and leftmost_right_x is not None
            new_horizontal_line = ((rightmost_left_x, y), (leftmost_right_x, y))
            existing_horizontal_line_endpoints.append(new_horizontal_line)
        else:
            # vertical line
            below_crossers = []
            above_crossers = []
            abovemost_below_y = None
            belowmost_above_y = None
            for p0, p1 in existing_horizontal_line_endpoints:
                # x is constant on the new line so we want to check their x ranges
                x0, y0 = p0
                x1, y1 = p1
                assert y0 == y1
                if x0 < x < x1:
                    # it's a crosser
                    if y0 < y:
                        below_crossers.append((p0, p1))
                        abovemost_below_y = max(abovemost_below_y, y0) if abovemost_below_y is not None else y0
                    elif y0 > y:
                        above_crossers.append((p0, p1))
                        belowmost_above_y = min(belowmost_above_y, y0) if belowmost_above_y is not None else y0
                    else:
                        # it can pass through this line
                        pass
            assert abovemost_below_y is not None and belowmost_above_y is not None
            new_vertical_line = ((x, abovemost_below_y), (x, belowmost_above_y))
            existing_vertical_line_endpoints.append(new_vertical_line)

    return existing_horizontal_line_endpoints + existing_vertical_line_endpoints



if __name__ == "__main__":
    n = 5000

    get_reference_points = get_points_positive_reinforcement

    points = get_reference_points(n)
    random.shuffle(points)
    xs = points[:,0]
    ys = points[:,1]
    plt.scatter(xs, ys)
    plt.gca().set_aspect("equal")
    plt.show()

    verticalities = get_line_directions(n)
    endpoints = get_line_segment_endpoints(points, verticalities)
    for (x0,y0), (x1,y1) in endpoints:
        plt.plot([x0, x1], [y0, y1], c="k")
    plt.gca().set_aspect("equal")
    plt.show()
