import random
import numpy as np
import matplotlib.pyplot as plt


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


class Line:
    def __init__(self, locations, stitch_type, color):
        assert all(type(x) is Point for x in locations)
        self.locations = locations
        assert Stitch.is_valid(stitch_type)
        self.stitch_type = stitch_type
        self.color = color

    @staticmethod
    def random():
        n_points = random.randint(20, 100)
        # locs = Line.get_random_path_completely_random(n_points)
        locs = Line.get_random_path_with_momentum(n_points, perturbation=0.1)
        return Line.random_from_locs(locs)

    @staticmethod
    def random_from_locs(locs):
        stitch_type = Stitch.random(1)
        color = get_random_color()
        return Line(locs, stitch_type, color)

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
    def get_random_closed_path(n_points, perturbation):
        # start with a circle and perturb the path
        # TODO: opposite_direction = random.choice([True, False])  # which way theta rotates for each path increment around the perimeter of the baseline circle
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

        # d_theta_left = 2*np.pi
        # d_theta_this_step = d_theta_left / n_steps_left
        # first_step_length = ((radius * np.cos(d_theta_this_step) - radius)**2 + (radius * np.sin(d_theta_this_step))**2) ** 0.5  # from initial point at (1, 0) to first point at (r cos theta, r sin theta)

        # now set p0 as the current point
        p = p0
        v = v0  # displacement vector from center
        for step_i in range(n_points):
            theta_left = angle_between_directional_2d(v, v0)  # FROM current point, about center, TO p0 (the first point we put on the edge)
            if theta_left == 0 and n_steps_left > 0:
                # the case where we have only just put p0 on the circle and haven't traversed any angle yet
                theta_left += 2*np.pi
            print(f"n_steps_left = {n_steps_left}; theta_left = {180/np.pi*theta_left} deg")

            # just get the next unperturbed point by getting a vector of same radius as vi but with incremented theta, rather than trying to actually do the math to rotate it about the center or anything like that
            next_d_theta = theta_left / n_steps_left
            r = mag_2d(v)  # current radius
            theta_position = angle_2d(v)
            next_theta_position = theta_position + next_d_theta
            v_next = np.array([r * np.cos(next_theta_position), r * np.sin(next_theta_position)])
            p_next = c + v_next
            print(f"- next d theta = {180/np.pi*next_d_theta} deg\n  r = {r}\n  theta_position = {180/np.pi*theta_position} deg\n  next_theta_position = {180/np.pi*next_theta_position} deg\n  v_next = {v_next}\n  p_next = {p_next}\n  c = {c}\n")
            while True:
                q = Point.random_2d_uniform(max_r=perturbation).to_cartesian_array()  # perturbation of this new point
                print(f"rolled perturbation: {q}")
                p_perturbed = p_next + q
                candidate_point = Point.from_cartesian(*p_perturbed)
                if candidate_point.r < 1:
                    # accept the point, it's within the boundaries of the hoop
                    p = p_perturbed
                    v = p_perturbed - c
                    points.append(candidate_point)
                    n_steps_left -= 1
                    print(f"accepted new point {candidate_point}, {n_steps_left} steps left")
                    break
                else:
                    print(f"perturbation failed at step {step_i}, rerolling")
        return points


class Patch:
    def __init__(self, border_line, stitch_type, color, show_border_line):
        # border locations treated as a path around the border, and we assume we will connect the end to the beginning
        assert type(border_line) is Line
        self.border_line = border_line
        assert Stitch.is_valid(stitch_type)
        self.stitch_type = stitch_type
        self.color = color
        assert type(show_border_line) is bool
        self.show_border_line = show_border_line

    @staticmethod
    def random():
        n_points = random.randint(20, 100)
        # n_points = 8  # debug
        perturbation = 0.1
        # perturbation = 0.0  # debug
        locs = Line.get_random_closed_path(n_points, perturbation=perturbation)
        border_line = Line.random_from_locs(locs)
        border_line.close_shape()
        stitch_type = Stitch.random(2)
        color = get_random_color()
        show_border_line = random.choice([True, False])
        return Patch(border_line, stitch_type, color, show_border_line)

    def get_xs_and_ys(self):
        return self.border_line.get_xs_and_ys()


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
        for patch in self.patches:
            self.plot_patch(patch)

        for line in self.lines:
            self.plot_line(line)

        for knot in self.knots:
            self.plot_knot(knot)

    # could be a static method but whatever
    def plot_patch(self, patch):
        xs, ys = patch.get_xs_and_ys()
        plt.fill(xs, ys, color=patch.color)
        # TODO stitch type texture

        if patch.show_border_line:
            self.plot_line(patch.border_line)

    def plot_line(self, line):
        xs, ys = line.get_xs_and_ys()
        plt.plot(xs, ys, color=line.color)
        # TODO stitch type texture

    def plot_knot(self, knot):
        x, y = knot.location.to_cartesian()
        color = knot.color
        marker = Stitch.get_marker(knot.knot_type)
        plt.scatter(x, y, color=color, marker=marker)


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



if __name__ == "__main__":
    hoop = Hoop()

    for i in range(5):
        hoop.add_knot(Knot.random())
        hoop.add_line(Line.random())
        hoop.add_patch(Patch.random())

    hoop.plot()
    plt.show()
