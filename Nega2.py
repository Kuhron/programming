import numpy as np
import matplotlib.pyplot as plt


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

    def direction_color(self):
        x_pos = np.array((1, 0, 0))
        x_neg = np.array((0, 1, 1))
        y_pos = np.array((0, 1, 0))
        y_neg = np.array((1, 0, 1))
        z_pos = np.array((0, 0, 1))
        z_neg = np.array((1, 1, 0))
        d = self.direction()
        x_c = abs(d.x) * (x_pos if d.x >= 0 else x_neg)
        y_c = abs(d.y) * (y_pos if d.y >= 0 else y_neg)
        z_c = abs(d.z) * (z_pos if d.z >= 0 else z_neg)
        return x_c + y_c + z_c

    @staticmethod
    def dot(v1, v2):
        return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z


class Grain:
    def __init__(self, location, moment):
        assert type(location) is Coordinates
        assert type(moment) is Vector
        self.location = location
        self.moment = moment

    def __repr__(self):
        return "Grain {} {}".format(self.location, self.moment)

    def get_effective_moment_due_to_grain_at_point(self, point):
        point_relative_coords = CoordinatesRelativeToOtherMoment.from_grain_and_other_point(self, point)
        r = point_relative_coords.r
        theta = point_relative_coords.theta

        direction_change = lambda v: Vector(v.x, v.y, v.z)  # maybe this will depend on r and theta somehow, but simplest is that the whole field due to a single grain points the same direction
        magnitude_factor = np.exp(-r/RADIUS_DECAY)
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

    def plot_field(self, x_min, x_max, y_min, y_max, fixed_z):
        x_steps = 100
        y_steps = 100
        xs = np.linspace(x_min, x_max, x_steps)
        ys = np.linspace(y_min, y_max, y_steps)

        field = self.get_field(xs, ys, fixed_z)
        field_magnitudes = grid_map(lambda g: g.moment.magnitude(), field)
        # field_directions = grid_map(lambda v: v.direction(), field)
        flat_field = flatten_grid(field)
        # field_xs = [v.x for v in flat_field]  # these are the xs of the VECTORS in the field, not their locations! makes cool plot if you scatter these instead of locations, though!
        field_xs = [g.location.x for g in flat_field]
        field_ys = [g.location.y for g in flat_field]
        direction_colors = [g.moment.direction_color() for g in flat_field]
        # print(field_magnitudes)

        plt.subplot(1, 2, 1)
        plt.imshow(field_magnitudes)
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.scatter(field_xs, field_ys, c=direction_colors)
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
        return Grain(point, total_moment)


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


if __name__ == "__main__":
    grains = [
        Grain(Coordinates(-1, -1, 0), Vector(1, 0, 0)),
        Grain(Coordinates(-1, 1, 0), Vector(-1, 0, 0)),
        Grain(Coordinates(1, -1, 0), Vector(-1, 0, 0)),
        Grain(Coordinates(1, 1, 0), Vector(1, 0, 0)),
    ]

    desert = Desert(grains)
    for fixed_z in [-1, 0, 1]:
        desert.plot_field(x_min=-5, x_max=5, y_min=-5, y_max=5, fixed_z=fixed_z)
