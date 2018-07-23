# procedure: start with a horizontal line segment, such as [0, 1].
# iteratively add a shape to each segment in the fractal, where the shape added
# - is composed of multiple segments
# - begins and ends at 0
# and then add its segment divisions to the set of segment divisions of the fractal
# for example, could start with a shape like _, then add either _/\_ or its negative (random choice),
# then the original segment becomes four segments, then repeat this process on each of those four segments
# (so on the slanted ones, would just add a scaled-down version of one of the _/\_ shapes), and repeat to arbitrary precision
# this will make a wave-like shape with various fluctuations
# other types of added shapes could be used, with other numbers of segments, could even have a set of choices with differing numbers of segments
# so try to make code general


import random

import numpy as np
import matplotlib.pyplot as plt


class Segment:
    def __init__(self, left_x, slope):
        self.left_x = left_x
        self.slope = slope


class Curve:
    # can use integration approach by just writing slopes that start at each endpoint, sum all previous ones, then add (x-last_endpoint)*(slope)
    def __init__(self, segments=None):
        self.segments = [] if segments is None else segments

    def add_segment(self, segment):
        self.segments.append(segment)

    def remove_segment(self, segment):
        self.segments.remove(segment)

    def sort_segments(self):
        left_xs = self.get_left_xs()
        assert len(set(left_xs)) == len(left_xs), "segments have duplicate starting points"
        self.segments = sorted(self.segments, key=lambda seg: seg.left_x)

    def get_left_xs(self):
        return [seg.left_x for seg in self.segments]

    def get_slopes(self):
        return [seg.slope for seg in self.segments]

    def integrate_previous_whole_segments(self, x):
        left_xs = self.get_left_xs()
        slopes = self.get_slopes()
        return Curve.static_integrate(left_xs, slopes, x)

    @staticmethod
    def static_integrate(left_xs, slopes, x_to_stop_at):
        x = x_to_stop_at
        result = 0
        for left_x, right_x, slope in zip(left_xs[:-1], left_xs[1:], slopes):
            if right_x > x:  # if it's equal, go ahead and add this segment, and get_value_at_point() should not double-count
                break
            result += (right_x - left_x) * slope
            # if x is the far right edge, iteration will stop and the whole derivative will have been integrated
        return result

    def get_value_at_point(self, x):
        self.sort_segments()
        left_xs = self.get_left_xs()
        slopes = self.get_slopes()
        assert x >= left_xs[0] and slopes[-1] is None and x <= left_xs[-1], "x is out of range, or you do not have a None slope for the right endpoint of the curve"
        index = max(i for i, left_x in enumerate(left_xs) if left_x <= x)
        last_point_less_than_equal = left_xs[index]
        slope = slopes[index]  # still just index, not index + 1, because each index is a left_x and then the slope to its right
        # if x is equal to a point, then the segment to its left will be counted by integrate_previous_whole_segments(), so there is no remainder
        remainder = x - last_point_less_than_equal
        residue = remainder * slope if remainder > 0 else 0  # avoid error where 0 * None raises
        return self.integrate_previous_whole_segments(x) + residue

    def add_fractal_step(self, left_x, fractal_step):
        self.sort_segments()
        left_xs = self.get_left_xs()
        assert left_xs[0] <= left_x < left_xs[-1], "{} <= {} < {}".format(left_xs[0], left_x, left_xs[-1])  # can't be equal to far right edge because then you have no segment to add slope to
        affected_segment_indexes = [i for i, seg in enumerate(self.segments) if seg.left_x == left_x]
        assert len(affected_segment_indexes) == 1
        seg = self.segments[affected_segment_indexes[0]]
        right_x = self.segments[affected_segment_indexes[0] + 1].left_x
        slope = seg.slope
        new_left_xs = [left_x + (x * (right_x - left_x)) for x in fractal_step.left_xs[:-1]]  # assumes fractal step domain is normalized to [0, 1]
        new_slopes = [slope + s for s in fractal_step.slopes[:-1]]
        self.remove_segment(seg)
        for left_x, slope in zip(new_left_xs, new_slopes):
            self.add_segment(Segment(left_x, slope))

    def iterate_fractal(self, fractal_step_choices):
        self.sort_segments()
        for left_x in self.get_left_xs()[:-1]:
            self.add_fractal_step(left_x, random.choice(fractal_step_choices))

    def plot(self, precision=0.01):
        xs = np.arange(0, 1 + precision, precision)
        ys = [self.get_value_at_point(x) for x in xs]
        plt.plot(xs, ys)
        plt.show()


class FractalStep:
    def __init__(self, left_xs, slopes):
        # make sure slope shape is normalized to domain of [0, 1] and integrates to 0 (do this in the constructor for those shapes itself)
        assert left_xs == sorted(left_xs), "sort your damn left xs"
        assert left_xs[0] == 0 and left_xs[-1] == 1, "domain must be [0, 1]"
        assert slopes[-1] is None, "need None slope for far right edge"
        integral = Curve.static_integrate(left_xs, slopes, left_xs[-1])
        assert abs(integral) < 1e-9, "must integrate to 0, but got {}".format(integral)

        self.left_xs = left_xs
        self.slopes = slopes


if __name__ == "__main__":
    initial_segments = [Segment(0, 0), Segment(1, None)]
    curve = Curve(initial_segments)

    long_fly_step  = FractalStep([0, 0.25, 0.5, 0.75, 1], [0, 1, -1, 0, None])
    short_fly_step = FractalStep([0, 0.25, 0.5, 0.75, 1], [0, -1, 1, 0, None])

    long_condor_step  = FractalStep([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 1, 0, -1, 0, None])
    short_condor_step = FractalStep([0, 0.2, 0.4, 0.6, 0.8, 1], [0, -1, 0, 1, 0, None])

    # choices = [long_fly_step, short_fly_step, long_condor_step, short_condor_step]
    choices = [long_condor_step]

    n_steps = 5
    for i in range(n_steps):
        curve.iterate_fractal(choices)

    curve.plot(precision=1e-3)




