import numpy as np
import matplotlib.pyplot as plt
import random


RADIUS_DECAY = 1

class Coordinates:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return "p({}, {}, {})".format(self.x, self.y, self.z)


class CoordinatesRelativeToOtherMoment:
    # where is this point relative to a reference moment in space?
    # the coordinate system satisfies:
    #   - reference moment is located at r=0, and it points up, 
    #       like a vector sticking up from the center of the galaxy perpendicular to the plane
    #   - theta = 0 means at the zenith, directly above the reference moment (i.e., in the direction it points)
    #   - theta = 90 deg means beside the reference moment (in the plane of the galaxy)
    #   - theta = 180 deg means at the nadir
    #   - the whole system is radially symmetric about the reference moment's vector as an axis, i.e., longitude doesn't matter

    def __init__(self, r, theta):
        self.r = r
        self.theta = theta

    @staticmethod
    def from_grain_and_other_point(grain, other_point):
        # set the grain at (0, 0, 0)
        dx = other_point.x - grain.location.x
        dy = other_point.y - grain.location.y
        dz = other_point.z - grain.location.z
        # get the theta
        displacement_of_other_point = Vector(dx, dy, dz)
        r = displacement_of_other_point.magnitude()
        if r == 0:
            return CoordinatesRelativeToOtherMoment(0, 0)
        moment = grain.moment
        v1 = displacement_of_other_point
        v2 = moment
        cos_theta = Vector.dot(v1, v2) / (v1.magnitude() * v2.magnitude())
        theta = np.arccos(cos_theta)
        assert 0 <= theta <= np.pi, theta
        return CoordinatesRelativeToOtherMoment(r, theta)


class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return "v({}, {}, {}, |{}|)".format(self.x, self.y, self.z, self.magnitude())

    def magnitude(self):
        return np.linalg.norm((self.x, self.y, self.z))

    def direction(self):
        m = self.magnitude()
        return Vector(self.x/m, self.y/m, self.z/m)

    def as_array(self):
        return np.array((self.x, self.y, self.z))

    def direction_color(self):
        x_pos = np.array((1, 0, 0))
        x_neg = np.array((0, 1, 1))
        y_pos = np.array((0, 1, 0))
        y_neg = np.array((1, 0, 1))
        z_pos = np.array((0, 0, 1))
        z_neg = np.array((1, 1, 0))
        # note that this color space is ambiguous (cyan could be (negative x) or (positive y plus positive z))
        # maybe I will make a better one later if the visualization of direction as color ends up useful enough to warrant it
        d = self.direction()
        x_c = abs(d.x) * (x_pos if d.x >= 0 else x_neg)
        y_c = abs(d.y) * (y_pos if d.y >= 0 else y_neg)
        z_c = abs(d.z) * (z_pos if d.z >= 0 else z_neg)
        return Vector(*(x_c + y_c + z_c)).direction().as_array()

    @staticmethod
    def dot(v1, v2):
        return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z

    @staticmethod
    def get_random_unit_vector():
        return Vector(*r(-1, 1, 3)).direction()


class Grain:
    def __init__(self, phonic, location, moment):
        assert type(location) is Coordinates
        assert type(moment) is Vector
        self.phonic = phonic
        self.location = location
        self.moment = moment

    def __repr__(self):
        return "Grain {} {}".format(self.location, self.moment)

    def get_effective_moment_due_to_grain_at_point(self, point):
        if self.phonic is None:
            raise Exception("trying to get effective field from pseudo-grain (not a phonic, just a moment field value at a point)")
        point_relative_coords = CoordinatesRelativeToOtherMoment.from_grain_and_other_point(self, point)
        r = point_relative_coords.r
        theta = point_relative_coords.theta

        direction_change = lambda v: Vector(v.x, v.y, v.z)  # maybe this will depend on r and theta somehow, but simplest is that the whole field due to a single grain points the same direction

        unitless_radius = r/RADIUS_DECAY
        magnitude_factor = np.exp(-unitless_radius)  # the radius decay constant is arbitrary, could be a parameter of the grain, but I don't want to add that extra parameter
        # magnitude_factor = 1/(unitless_radius**2)

        if self.phonic == "positic":
            magnitude_factor *= positic_magnitude_factor(theta)
        elif self.phonic == "negatic":
            magnitude_factor *= negatic_magnitude_factor(theta)
        elif self.phonic == "zerotic":
            magnitude_factor *= zerotic_magnitude_factor(theta)
        else:
            raise ValueError("unknown phonic type {}".format(self.phonic))

        # print("r {} gave magfac {}".format(r, magnitude_factor))
        res = direction_change(self.moment)
        # print("changed direction of {} to {}".format(self.moment, res))
        res.x *= magnitude_factor
        res.y *= magnitude_factor
        res.z *= magnitude_factor
        # print("g {} ; p {} ; res {}".format(self, point, res))
        return res


class Desert:
    def __init__(self, grains):
        self.grains = grains

    def plot_field(self, x_min, x_max, y_min, y_max, fixed_z, resolution_steps=100):
        xs = np.linspace(x_min, x_max, resolution_steps)
        ys = np.linspace(y_min, y_max, resolution_steps)

        field = self.get_field(xs, ys, fixed_z)
        field_magnitudes = grid_map(lambda g: g.moment.magnitude(), field)
        max_mag = max(max(row) for row in field_magnitudes)
        max_mag = 1 if max_mag == 0 else max_mag  # fury road
        direction_colors_grid = grid_map(lambda g: g.moment.direction_color(), field)
        magnitude_color_product_grid = [[x/max_mag for x in row] for row in elementwise_product(field_magnitudes, direction_colors_grid)]
        flat_field = flatten_grid(field)
        # field_xs = [v.x for v in flat_field]  # these are the xs of the VECTORS in the field, not their locations! makes cool plot if you scatter these instead of locations, though!
        field_xs = [g.location.x for g in flat_field]
        field_ys = [g.location.y for g in flat_field]
        # direction_colors = [g.moment.direction_color() for g in flat_field]
        # direction_colors = flatten_grid(direction_colors_grid)
        # print(field_magnitudes)

        plt.subplot(1, 3, 1)
        plt.imshow(np.array(field_magnitudes).T, origin="lower")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.scatter(field_xs, field_ys, c=flatten_grid(magnitude_color_product_grid))
        plt.axis("equal")

        plt.subplot(1, 3, 3)
        plt.scatter(field_xs, field_ys, c=flatten_grid(direction_colors_grid))
        plt.axis("equal")
        plt.show()


    def get_field(self, xs, ys, fixed_z):
        def f(x, y):
            return self.get_field_at_point(Coordinates(x, y, fixed_z))

        return cartesian_product_map(f, xs, ys)

    def get_field_at_point(self, point):
        moments_due_to_each_grain = []
        for g in self.grains:
            contribution = g.get_effective_moment_due_to_grain_at_point(point)
            moments_due_to_each_grain.append(contribution)
        total_moment = MomentMath.get_total_moment(moments_due_to_each_grain)
        # print("TOTAL {}".format(total_moment))
        return Grain(None, point, total_moment)  # pseudo-grain, not a phonic, just a moment field value at a point


class MomentMath:
    @staticmethod
    def get_total_moment(moments):
        return MomentMath.sum_moments(moments)  # simple, but probably too boring

    @staticmethod
    def sum_moments(moments):
        res = Vector(0, 0, 0)
        for m in moments:
            res.x += m.x
            res.y += m.y
            res.z += m.z
        # print("summing moments:\n{}\ngave {}\n".format(moments, res))
        return res


def flatten_grid(grid):
    if type(grid) is list:
        # assert type(grid[0]) is not list
        i_lim = len(grid)
        j_lim = len(grid[0])
    else:
        assert (hasattr(grid, "shape") and len(grid.shape) == 2)
        i_lim, j_lim = grid.shape
    res = []
    for i in range(i_lim):
        for j in range(j_lim):
            res.append(grid[i][j])
    return res


def cartesian_product_map(f, xs, ys):
    # return f(xs[:,None], ys[None,:])
    res = [[None for j in range(len(ys))] for i in range(len(xs))]  # fuck figuring out this syntax
    for x_i in range(len(xs)):
        for y_i in range(len(ys)):
            res[x_i][y_i] = f(xs[x_i], ys[y_i])
    return res


def grid_map(f, grid):
    grid = np.array(grid)
    assert len(grid.shape) == 2
    res = [[None for j in range(grid.shape[1])] for i in range(grid.shape[0])]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            res[i][j] = f(grid[i][j])
    return res


def elementwise_product(grid1, grid2):
    x1 = len(grid1)
    x2 = len(grid2)
    y1 = len(grid1[0])
    y2 = len(grid2[0])
    assert x1 == x2 and y1 == y2
    res = [[None for j in range(y1)] for i in range(x1)]
    for i in range(x1):
        for j in range(x2):
            res[i][j] = grid1[i][j] * grid2[i][j]
    return res


def r(a, b, n):
    return np.random.uniform(a, b, n)


def positic_magnitude_factor(theta):
    # return np.sin(abs(theta))
    return np.cos(abs(theta))**2

def negatic_magnitude_factor(theta):
    # return 1 - np.sin(abs(theta))
    return np.sin(abs(theta))**2
    # note that sin^2 and cos^2 look the same in the cross section, but the sin^2 is a donut and the cos^2 is two balls
    # I want nega to look like a donut

def zerotic_magnitude_factor(theta):
    return 1

def plot_magnitude_factor_functions():
    thetas = np.linspace(-np.pi, np.pi, 500)
    pos_rs = positic_magnitude_factor(thetas)
    neg_rs = negatic_magnitude_factor(thetas)
    zer_rs = zerotic_magnitude_factor(thetas)
    plt.polar(thetas, pos_rs, "b")
    plt.polar(thetas, neg_rs, "r")
    # don't worry about plotting zerotic
    plt.show()


def choose_phonic():
    # return "positic"
    # return "negatic"
    # return "zerotic"
    return random.choice(["positic", "negatic", "zerotic"])


if __name__ == "__main__":
    # plot_magnitude_factor_functions()

    grains = [
        # Grain("positic", Coordinates(0, -1, 0), Vector(0, 1, 0)),
        # Grain("positic", Coordinates(0, 1, 0), Vector(0, -1, 0)),
        # Grain("positic", Coordinates(-1, 0, 0), Vector(1, 0, 0)),
        # Grain("positic", Coordinates(1, 0, 0), Vector(-1, 0, 0)),

        Grain("positic", Coordinates(-np.sin(t), np.cos(t), 0), Vector(-np.cos(t), -np.sin(t), 0)) for t in np.arange(0, 2*np.pi, 2*np.pi/36)
        # Grain("negatic", Coordinates(np.cos(t), np.sin(t), 0), Vector(-np.cos(t), -np.sin(t), 0)) for t in np.arange(0, 2*np.pi, 2*np.pi/36)

        # Grain(choose_phonic(), Coordinates(r(-10,10,1), r(-10,10,1), r(-0.1,0.1,1)), Vector.get_random_unit_vector()) for _ in range(100)  # all mag 1, random directions
        # Grain(Coordinates(r(-4,4,1), r(-4,4,1), 0), Vector(0, 0, 1)) for _ in range(100)  # all mag 1, same direction
    ]

    desert = Desert(grains)
    for fixed_z in [0]:
        desert.plot_field(x_min=-3, x_max=3, y_min=-3, y_max=3, fixed_z=fixed_z, resolution_steps=100)
