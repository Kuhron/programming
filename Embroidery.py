import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


# the canvas is the unit circle


class Point:
    def __init__(self, r, theta):
        self.r = r
        self.theta = theta
        self.theta_degrees = 180/np.pi * theta
        self.x = Point.get_x(r, theta)
        self.y = Point.get_y(r, theta)

    @staticmethod
    def from_polar(r, theta):
        return Point(r, theta)

    @staticmethod
    def from_cartesian(x, y):
        r = Point.get_r(x, y)
        theta = Point.get_theta(x, y)
        return Point(r, theta)

    @staticmethod
    def get_x(r, theta):
        return r * np.cos(theta)

    @staticmethod
    def get_y(r, theta):
        return r * np.sin(theta)

    @staticmethod
    def get_r(x, y):
        return (x**2 + y**2) ** 0.5

    @staticmethod
    def get_theta(x, y):
        return np.arctan2(y, x)

    def __eq__(self, other):
        if type(other) is not Point:
            return NotImplemented
        return self.r == other.r and Point.thetas_equal(self.theta, other.theta)

    @staticmethod
    def thetas_equal(t1, t2):
        return t1 % (2*np.pi) == t2 % (2*np.pi)

    @staticmethod
    def random_2d_uniform(max_r=1):
        # do the thing where we get uniform distribution by generating in 2d box and throwing out points outside of the circle
        while True:
            x,y = np.random.uniform(-max_r, max_r, (2,))
            r = Point.get_r(x, y)
            if r <= max_r:
                return Point.from_cartesian(x, y)

    @staticmethod
    def random_center_bias(max_r=1):
        r = random.uniform(0, max_r)
        theta = random.uniform(0, 2*np.pi)
        return Point.from_polar(r, theta)

    def to_cartesian(self):
        return self.x, self.y

    def to_cartesian_array(self):
        return np.array([self.x, self.y])

    def to_polar(self):
        return self.r, self.theta

    def to_polar_array(self):
        return np.array([self.r, self.theta])

    def rotated_about_center(self, d_theta, deg=False):
        if deg:  # theta was passed in degrees, convert it to radians
            d_theta = np.pi/180 * d_theta
        r, theta = self.to_polar()
        theta += d_theta
        return Point.from_polar(r, theta)

    def dilated(self, r_factor):
        r, theta = self.to_polar()
        r *= r_factor
        return Point.from_polar(r, theta)

    def dilated_to_r(self, target_r):
        r, theta = self.to_polar()
        return Point.from_polar(target_r, theta)

    def __repr__(self):
        return f"<Point r={self.r:.4f}, theta={self.theta_degrees:.2f} deg, x={self.x:.4f}, y={self.y:.4f}>"


class Stitch:
    STITCH_TYPES_0D = ["french", "colonial", "lazy daisy"]
    STITCH_TYPES_1D = ["straight", "stem", "running", "back", "chain"]
    STITCH_TYPES_2D = ["seed", "satin", "padded satin", "fishbone", "long and short"]
    ALL_STITCH_TYPES = STITCH_TYPES_0D + STITCH_TYPES_1D + STITCH_TYPES_2D

    MARKERS_0D = {"french": ".", "colonial": "v", "lazy daisy": "d"}

    @staticmethod
    def get_dimensionality(stitch_type):
        if stitch_type in Stitch.STITCH_TYPES_0D:
            return 0
        elif stitch_type in Stitch.STITCH_TYPES_1D:
            return 1
        elif stitch_type in Stitch.STITCH_TYPES_2D:
            return 2
        else:
            raise ValueError(f"unknown stitch: {stitch_type}")

    @staticmethod
    def is_valid(stitch_type):
        return stitch_type in Stitch.ALL_STITCH_TYPES

    @staticmethod
    def random(dimensionality):
        if dimensionality == 0:
            return random.choice(Stitch.STITCH_TYPES_0D)
        elif dimensionality == 1:
            return random.choice(Stitch.STITCH_TYPES_1D)
        elif dimensionality == 2:
            return random.choice(Stitch.STITCH_TYPES_2D)
        else:
            raise ValueError(f"invalid dimensionality: {dimensionality}")

    @staticmethod
    def get_marker(stitch_type):
        dim = Stitch.get_dimensionality(stitch_type)
        if dim == 0:
            return Stitch.MARKERS_0D[stitch_type]
        else:
            raise NotImplementedError


class Knot:
    def __init__(self, location, knot_type, color):
        assert type(location) is Point
        self.location = location
        assert knot_type in Stitch.STITCH_TYPES_0D
        self.knot_type = knot_type
        self.color = color

    @staticmethod
    def random():
        loc = Point.random_2d_uniform()
        knot_type = Stitch.random(0)
        color = get_random_color()
        return Knot(loc, knot_type, color)

    def rotated_about_center(self, d_theta, deg=False):
        new_location = self.location.rotated_about_center(d_theta, deg=deg)
        return Knot(new_location, self.knot_type, self.color)

    def dilated(self, r_factor):
        new_location = self.location.dilated(r_factor)
        return Knot(new_location, self.knot_type, self.color)

    def radius_limited(self, max_r):
        new_location_xy, = limit_r_of_points([self.location], max_r=1, as_array=False)
        new_location = Point(*new_location_xy)
        return Knot(new_location, self.knot_type, self.color)


class Line:
    def __init__(self, locations, stitch_type, color, smoothing):
        assert all(type(x) is Point for x in locations)
        self.locations = locations
        assert Stitch.is_valid(stitch_type)
        self.stitch_type = stitch_type
        self.color = color
        self.smoothing = smoothing

    @staticmethod
    def random(n_points=None, perturbation=0.1):
        if n_points is None:
            n_points = random.randint(20, 100)
        # locs = Line.get_random_path_completely_random(n_points)
        locs = Line.get_random_path_with_momentum(n_points, perturbation=perturbation)
        return Line.random_from_locs(locs)

    @staticmethod
    def random_from_locs(locs):
        stitch_type = Stitch.random(1)
        color = get_random_color()
        smoothing = abs(np.random.normal(0, 1))
        return Line(locs, stitch_type, color, smoothing)

    def close_shape(self):
        if self.locations[0] == self.locations[-1]:
            print("shape already closed")
            return
        else:
            self.locations.append(self.locations[0])

    def get_xs_and_ys(self):
        xs = []
        ys = []
        for loc in self.locations:
            xs.append(loc.x)
            ys.append(loc.y)
        return xs, ys

    def get_xs_and_ys_arrays(self):
        xs, ys = self.get_xs_and_ys()
        xs = np.array(xs)
        ys = np.array(ys)
        return xs, ys

    def get_xs_and_ys_smoothed(self, smoothing=0, resolution=100, closed=False):
        # https://stackoverflow.com/questions/53328619/smooth-the-path-of-line-with-python
        if closed:
            self.close_shape()
        x, y = self.get_xs_and_ys_arrays()
        assert len(x) == len(y)
        assert x.ndim == 1 and y.ndim == 1
        if len(x) == 2:
            # too few points for spline interpolation
            return x, y
        elif len(x) == 3:
            # too few points for cubic spline (which is default)
            spline_degree = 2
        else:
            spline_degree = 3
        # print("x=",x)
        # print("y=",y)
        # x = np.r_[x, x[0]]  # this is adding the first element to the end again
        # y = np.r_[y, y[0]]  # this is adding the first element to the end again
        f, u = interpolate.splprep([x, y], s=smoothing, per=False, k=spline_degree)  # per=True means it treats the input as periodic, but I am letting user choose whether the shape is closed (i.e. treated as periodic) or not
        # create interpolated lists of points
        xint, yint = interpolate.splev(np.linspace(0, 1, resolution), f)

        # the interpolated path may exit the hoop, fix that
        arr = np.array([xint, yint])
        points = arr.T
        points = limit_r_of_points(points, max_r=1)
        xint, yint = points.T
        return xint, yint

    @staticmethod
    def get_random_path_completely_random(n_points):
        return [Point.random_2d_uniform() for i in range(n_points)]

    @staticmethod
    def get_random_path_with_momentum(n_points, perturbation):
        assert 0 <= perturbation < 1
        points = []
        starting_point = Point.random_center_bias()
        points.append(starting_point)

        p = starting_point.to_cartesian_array()
        target_p = Point.random_2d_uniform().to_cartesian_array()
        n_steps_left = n_points - 1  # fencepost
        while len(points) < n_points:
            velocity = (target_p - p) / (n_steps_left)
            velocity_perturbation = Point.random_2d_uniform(max_r=perturbation).to_cartesian_array()
            velocity += velocity_perturbation
            p += velocity
            point = Point.from_cartesian(*p)
            print("new point in path with momentum:", point)
            if point.r < 1:
                points.append(point)
                n_steps_left -= 1
        return points

    @staticmethod
    def get_random_closed_path(n_points, perturbation, self_intersection_tolerance=0.01):
        # start with a circle and perturb the path
        opposite_direction = random.choice([True, False])  # which way theta rotates for each path increment around the perimeter of the baseline circle
        # re-calculate what the rest of the circle should look like at each step after perturbation, based on current location and number of steps left

        points = []
        center_point = Point.random_2d_uniform()
        n_steps_left = n_points

        c = center_point.to_cartesian_array()
        r_center = center_point.r
        max_circle_radius = 1 - r_center  # don't let the circle go outside the hoop

        # get initial point away from center
        while True:
            r0 = random.uniform(0, max_circle_radius)  # can tweak this if necessary, it's only for getting the first vector
            theta0 = random.uniform(0, 2*np.pi)
            v0 = np.array([r0 * np.cos(theta0), r0 * np.sin(theta0)])
            p0 = c + v0
            new_point = Point.from_cartesian(*p0)
            if new_point.r < 1:
                points.append(new_point)
                break
                # n_steps_left is still the number of points since you have to come all the way back around (e.g. imagine n = 4, we've just put the first point on the edge of the circle, still need 4 segments to complete the trip around)
            else:
                print(f"failed to get p0 within hoop, got {new_point}")

        # now set p0 as the current point
        p = p0
        v = v0  # displacement vector from center of circle patch (NOT from center of hoop)
        for step_i in range(n_points):
            theta_left = angle_between_directional_2d(v, v0)  # FROM current point, about center, TO p0 (the first point we put on the edge)
            if theta_left == 0 and n_steps_left > 0:
                # the case where we have only just put p0 on the circle and haven't traversed any angle yet
                theta_left += 2*np.pi
            if opposite_direction:
                theta_left *= -1
            print(f"n_steps_left = {n_steps_left}; theta_left = {180/np.pi*theta_left} deg")

            # just get the next unperturbed point by getting a vector of same radius as v but with incremented theta, rather than trying to actually do the math to rotate it about the center or anything like that
            next_d_theta = theta_left / n_steps_left
            r = mag_2d(v)  # current radius
            theta_position = angle_2d(v)
            next_theta_position = theta_position + next_d_theta
            while True:
                v_next = np.array([r * np.cos(next_theta_position), r * np.sin(next_theta_position)])
                p_next = c + v_next
                # print(f"- next d theta = {180/np.pi*next_d_theta} deg\n  r = {r}\n  theta_position = {180/np.pi*theta_position} deg\n  next_theta_position = {180/np.pi*next_theta_position} deg\n  v_next = {v_next}\n  p_next = {p_next}\n  c = {c}\n")

                q = Point.random_2d_uniform(max_r=perturbation).to_cartesian_array()  # perturbation of this new point
                # print(f"rolled perturbation: {q}")
                p_perturbed = p_next + q
                candidate_point = Point.from_cartesian(*p_perturbed)
                last_point = points[-1]
                candidate_segment = (last_point, candidate_point)
                existing_path = points

                meets_r_requirement = candidate_point.r < 1
                if not meets_r_requirement:
                    print(f"perturbation failed at step {step_i} due to going beyond the hoop's border; rerolling")
                    r *= 1/2  # lazy way to put it back in the hoop limits when the unperturbed point is outside
                    continue

                intersects_self = segment_intersects_path(candidate_segment, existing_path)
                meets_self_intersection_requirement = (not intersects_self) or random.random() < self_intersection_tolerance
                if not meets_self_intersection_requirement:
                    print(f"perturbation failed at step {step_i} due to self-intersection; rerolling")
                    continue

                # point meets the requirements, accept it
                p = p_perturbed
                v = p_perturbed - c
                points.append(candidate_point)
                n_steps_left -= 1
                print(f"accepted new point {candidate_point}, {n_steps_left} steps left")
                break

        return points

    def rotated_about_center(self, d_theta, deg=False):
        new_locations = [loc.rotated_about_center(d_theta, deg=deg) for loc in self.locations]
        return Line(new_locations, self.stitch_type, self.color, self.smoothing)

    def dilated(self, r_factor):
        new_locations = [loc.dilated(r_factor) for loc in self.locations]
        return Line(new_locations, self.stitch_type, self.color, self.smoothing)

    def radius_limited(self, max_r):
        new_locations_xy = limit_r_of_points(self.locations, max_r=1, as_array=False)
        new_locations = [Point(*new_location_xy) for new_location_xy in new_locations_xy]
        return Line(new_locations, self.stitch_type, self.color, self.smoothing)



class Patch:
    def __init__(self, border_line, stitch_type, color, show_border_line, smoothing):
        # border locations treated as a path around the border, and we assume we will connect the end to the beginning
        assert type(border_line) is Line
        self.border_line = border_line
        assert Stitch.is_valid(stitch_type)
        self.stitch_type = stitch_type
        self.color = color
        assert type(show_border_line) is bool
        self.show_border_line = show_border_line
        self.smoothing = smoothing

    @staticmethod
    def random(n_points=None, perturbation=0.1):
        if n_points is None:
            n_points = random.randint(20, 100)
        # n_points = 8  # debug
        # perturbation = 0.0  # debug

        locs = Line.get_random_closed_path(n_points, perturbation=perturbation)
        border_line = Line.random_from_locs(locs)
        border_line.close_shape()
        stitch_type = Stitch.random(2)
        color = get_random_color()
        show_border_line = random.choice([True, False])
        smoothing = abs(np.random.normal(0, 1))
        return Patch(border_line, stitch_type, color, show_border_line, smoothing)

    def get_xs_and_ys(self):
        return self.border_line.get_xs_and_ys()

    def get_xs_and_ys_smoothed(self, smoothing=0, resolution=100):
        return self.border_line.get_xs_and_ys_smoothed(smoothing=smoothing, resolution=resolution, closed=True)

    def rotated_about_center(self, d_theta, deg=False):
        new_border_line = self.border_line.rotated_about_center(d_theta, deg=deg)
        return Patch(new_border_line, self.stitch_type, self.color, self.show_border_line, self.smoothing)

    def dilated(self, r_factor):
        new_border_line = self.border_line.dilated(r_factor)
        return Patch(new_border_line, self.stitch_type, self.color, self.show_border_line, self.smoothing)

    def radius_limited(self, max_r):
        new_border_line = self.border_line.radius_limited(max_r)
        return Patch(new_border_line, self.stitch_type, self.color, self.show_border_line, self.smoothing)



class Hoop:
    def __init__(self):
        self.knots = []  # 0 dimensional stitches
        self.lines = []  # 1 dimensional stitches
        self.patches = []  # 2 dimensional stitches

    def add_knot(self, knot):
        assert type(knot) is Knot
        assert knot.location.r < 1  # don't allow things right on the edge
        self.knots.append(knot)

    def add_line(self, line):
        assert type(line) is Line
        assert all(loc.r < 1 for loc in line.locations)
        self.lines.append(line)

    def add_patch(self, patch):
        assert type(patch) is Patch
        assert all(loc.r < 1 for loc in patch.border_line.locations)
        self.patches.append(patch)

    def plot(self):
        circle = plt.Circle((0, 0), 1, color="k", fill=False)
        plt.gca().add_patch(circle)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.gca().set_aspect('equal')

        # do larger dimensionality first, so patches don't cover lines and lines don't cover points
        zorder = 0
        for patch in self.patches:
            self.plot_patch(patch, zorder=zorder)
            zorder += 1

        for line in self.lines:
            self.plot_line(line, zorder=zorder)
            zorder += 1

        for knot in self.knots:
            self.plot_knot(knot, zorder=zorder)
            zorder += 1

    # could be a static method but whatever
    def plot_patch(self, patch, **kwargs):
        # xs, ys = patch.get_xs_and_ys()
        xs, ys = patch.get_xs_and_ys_smoothed(smoothing=patch.smoothing, resolution=100)
        plt.fill(xs, ys, color=patch.color, **kwargs)
        # TODO stitch type texture

        if patch.show_border_line:
            self.plot_line(patch.border_line, **kwargs)

    def plot_line(self, line, **kwargs):
        # xs, ys = line.get_xs_and_ys()
        xs, ys = line.get_xs_and_ys_smoothed(smoothing=line.smoothing, resolution=100)
        plt.plot(xs, ys, color=line.color, **kwargs)
        # TODO stitch type texture

    def plot_knot(self, knot, **kwargs):
        x, y = knot.location.to_cartesian()
        color = knot.color
        marker = Stitch.get_marker(knot.knot_type)
        plt.scatter(x, y, color=color, marker=marker, **kwargs)


def get_random_color():
    return np.random.rand(3)


def dot_2d(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    return x1 * y1 + x2 * y2


def mag_2d(v):
    x, y = v
    return (x**2 + y**2) ** 0.5


def angle_2d(v):
    x, y = v
    return np.arctan2(y, x)


def angle_between_directional_2d(v1, v2):
    # angle FROM v1 TO v2, in 0 <= theta < 2 pi
    # so order matters here!
    theta1 = angle_2d(v1)
    theta2 = angle_2d(v2)
    # going from 1 to 2, get theta2 - theta1
    res = (theta2 - theta1) % (2*np.pi)

    print(f"- v1 = {v1} with theta1 = {180/np.pi*theta1} deg\n  v2 = {v2} with theta2 = {180/np.pi*theta2} deg\n  angle from v1 to v2 = {180/np.pi*res} deg\n")
    return res


def segment_intersects_path(candidate_segment, existing_path):
    a, b = candidate_segment
    assert type(a) is Point
    assert type(b) is Point
    assert all(type(p) is Point for p in existing_path)
    for p, q in zip(existing_path[:-1], existing_path[1:]):
        # see if segment a-b intersects p-q
        # if it's the most recent segment, we expect that q (the most recent point) will be the same as our segment a (when adding to a path head-to-tail like we do in the line generation)
        result_for_common_endpoint = False if q == a else True  # we set intersection to False for the case where it's the head-tail linkage
        if segment_intersects_segment((a,b), (p,q), result_for_common_endpoint):
            # print(f"segments intersect:\n  a = {a}\n  b = {b}\n  p = {p}\n  q = {q}\n")
            return True
    return False


def segment_intersects_segment(seg1, seg2, result_for_common_endpoint=False):
    a, b = seg1
    p, q = seg2

    # ignore zero division for now, hope that it just never comes up because we're doing random directions
    xa, ya = a.to_cartesian()
    xb, yb = b.to_cartesian()
    xp, yp = p.to_cartesian()
    xq, yq = q.to_cartesian()
    
    # alpha is amount along the (b-a) vector starting from a
    # beta is amount along the (q-p) vector starting from p
    alpha = ((ya-yp)*(xq-xp)-(xa-xp)*(yq-yp)) / ((xb-xa)*(yq-yp)-(yb-ya)*(xq-xp))
    beta = ((yp-ya)*(xb-xa)-(xp-xa)*(yb-ya)) / ((xq-xp)*(yb-ya)-(yq-yp)*(xb-xa))

    raw_intersects = 0 <= alpha <= 1 and 0 <= beta <= 1
    has_common_endpoint = alpha in [0, 1] and beta in [0, 1]
    if has_common_endpoint:
        assert raw_intersects
        return result_for_common_endpoint
    else:
        return raw_intersects


def limit_r_of_points(points, max_r=1, as_array=True):
    res = []
    for item in points:
        if type(item) is Point:
            x, y = item.to_cartesian()
        else:
            x, y = item
        p = Point.from_cartesian(x, y)
        if p.r > max_r:
            new_r = min(max_r, p.r)
            new_p = p.dilated_to_r(new_r)
            print(f"limiting radius of point to {max_r}:\n   {p}\n-> {new_p}\n")
            input("check")
            res.append(new_p.to_cartesian())
        else:
            # it's fine as is
            res.append([x, y])

    if as_array:
        return np.array(res)
    else:
        return res


if __name__ == "__main__":
    hoop = Hoop()

    radial_symmetry = 1 if random.random() < 0.5 else random.randint(2, 8)
    dilation = random.uniform(0.5, 1.1)
    d_theta = 2*np.pi / radial_symmetry

    # n_knots = 50
    n_knots = random.randint(8, 50)

    # n_lines = 3
    n_lines = random.randint(3, 8)

    # n_patches = 3
    n_patches = random.randint(3, 6)

    for i in range(n_knots):
        k0 = Knot.random()
        for j in range(radial_symmetry):
            k = k0.rotated_about_center(j*d_theta)
            k = k.dilated(dilation**j)
            k = k.radius_limited(1)
            hoop.add_knot(k)

    for i in range(n_lines):
        n_points = random.randint(2, 30)
        perturbation = random.uniform(0, 0.15)
        l0 = Line.random(n_points=n_points, perturbation=perturbation)
        for j in range(radial_symmetry):
            l = l0.rotated_about_center(j*d_theta)
            l = l.dilated(dilation**j)
            # l = l.radius_limited(1)
            hoop.add_line(l)

    for i in range(n_patches):
        n_points = random.randint(6, 100)
        perturbation = random.uniform(0, 0.15)
        m0 = Patch.random(n_points=n_points, perturbation=perturbation)
        for j in range(radial_symmetry):
            m = m0.rotated_about_center(j*d_theta)
            m = m.dilated(dilation**j)
            # m = m.radius_limited(1)
            hoop.add_patch(m)

    hoop.plot()
    plt.show()
