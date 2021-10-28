import numpy as np
import matplotlib.pyplot as plt
import random
import sys


class Mapping:
    def get_convergence_points(self, x_min=-1, x_max=1, y_min=-1, y_max=1, resolution=100, max_iterations=10000, max_abs=1e6):
        print("getting convergence array")
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        convergence = np.zeros((len(xs), len(ys)))
        for ix, x in enumerate(xs):
            print(f"row {ix} of {len(xs)}")
            for iy, y in enumerate(ys):
                starting_point = (x, y)
                trajectory = get_late_trajectory(mapping, x, y, min_index=max_iterations-1, max_index=max_iterations, max_abs=max_abs)
                if trajectory_diverges(trajectory):
                    convergence[ix, iy] = 0
                else:
                    convergence[ix, iy] = 1
        print("done getting convergence array")
        return convergence

    def plot_convergence_points(self, x_min=-1, x_max=1, y_min=-1, y_max=1, resolution=100, max_iterations=10000, max_abs=1e6):
        convergence_array = self.get_convergence_points(x_min, x_max, y_min, y_max, resolution, max_iterations, max_abs)
        plt.imshow(convergence_array)
        plt.colorbar()
        plt.savefig("out_convergence.png")


class SingleMappingFromXY(Mapping):
    # maps a pair of coords to a single new value
    def __init__(self, coefficients):
        coefficients = np.array(coefficients)
        n_x_powers, n_y_powers = coefficients.shape
        self.x_powers, self.y_powers = np.meshgrid(range(n_x_powers), range(n_y_powers))
        self.coefficients = coefficients

    def __call__(self, x, y):
        x_power_matrix = x ** self.x_powers
        y_power_matrix = y ** self.y_powers
        monomial_values = self.coefficients * x_power_matrix * y_power_matrix
        return monomial_values.sum()


class DoubleMappingFromXY(Mapping):
    # maps a pair of coords to a pair of new values, for mapping from (x,y) at one timestep to the next
    def __init__(self, coefficients_x, coefficients_y):
        self.mapping_x = SingleMappingFromXY(coefficients_x)
        self.mapping_y = SingleMappingFromXY(coefficients_y)

    def __call__(self, x, y):
        new_x = self.mapping_x(x, y)
        new_y = self.mapping_y(x, y)
        return (new_x, new_y)

    @staticmethod
    def random(mag=1, max_degree=3):
        ndim = max_degree + 1  # e.g. for quadratic map, no term should have degree higher than 2, so want degrees 0, 1, and 2
        cx = np.random.uniform(-mag, mag, (ndim, ndim))
        cy = np.random.uniform(-mag, mag, (ndim, ndim))
        # both mappings should have lower triangle (where degrees are too large) set to zero coefficients
        # whole first row and column should be nonzero
        x_powers, y_powers = np.meshgrid(range(ndim), range(ndim))
        degree = x_powers + y_powers
        cx[degree > max_degree] = 0
        cy[degree > max_degree] = 0
        return DoubleMappingFromXY(cx, cy)

    @staticmethod
    def find_attractor(mag=1, max_degree=3):
        while True:
            mapping = DoubleMappingFromXY.random(mag, max_degree)
            x,y = np.random.uniform(-1, 1, 2)
            scatter_points = get_late_trajectory(mapping, x, y, min_index=1000, max_index=10000)
            if trajectory_diverges(scatter_points):
                print("divergent")
                continue
            n_unique_points = len(set(scatter_points))
            unique_ratio = n_unique_points / len(scatter_points)
            if unique_ratio < 0.1:
                print(f"not enough unique points: {n_unique_points}")
                continue
            return mapping

    def __repr__(self):
        cx_list = list(list(x) for x in self.mapping_x.coefficients)
        cy_list = list(list(y) for y in self.mapping_y.coefficients)
        return f"<Mapping XY with coefficients:\n\n{[cx_list, cy_list]}\n>"


class RandomChoiceMappingFromXY(Mapping):
    def __init__(self, mappings, probabilities):
        self.mappings = mappings
        p_total = sum(probabilities)
        self.probabilities = np.array(probabilities) / p_total

    def __call__(self, x, y):
        chosen_mapping = np.random.choice(self.mappings, p=self.probabilities)
        return chosen_mapping(x, y)


def get_late_trajectory(mapping, x0, y0, min_index, max_index, max_abs=1e9):
    assert max_index > min_index
    res = []
    x,y = x0,y0
    BIG = max_abs  # if it gets to this distance then assume it diverges
    for i in range(max_index):
        x,y = mapping(x,y)
        if i < min_index:
            # don't store these, let it settle into attractor state, if any
            continue
        else:
            res.append((x,y))
        if not np.isfinite(x) or not np.isfinite(y) or abs(x) > BIG or abs(y) > BIG:
            # assume it will diverge forever if it ever reaches inf/NaN
            # just put a nonfinite pair at the end and return
            res.append((np.inf, np.inf))
            return res
    return res


def trajectory_diverges(trajectory):
    x,y = trajectory[-1]
    return not np.isfinite(x) or not np.isfinite(y)


class MappingRecord:
    Ax, Ay = [[[-0.61812953, 0.62618444, 0.95847548], [-0.28566655, 0.60419982, 0], [-0.82349836, 0, 0]], [[0.41322135, -0.056517, 0.09507931], [0.84468446, -0.05060947, 0], [-0.9864845, 0, 0]]]
    Bx, By = [[[0.011100313095467884, -0.2653760753248544, 0.7175056845186025], [-0.6633260488831336, 0.16221771794919104, 0.0], [-0.6300085491546239, 0.0, 0.0]], [[0.9796117830195128, -0.8743965549320505, -0.9644515017383146], [0.12238901385759271, 0.3614148282517575, 0.0], [0.3019429543542915, 0.0, 0.0]]]
    Cx, Cy = [[[-0.4263134430524571, -0.2274150264834227, 0.07281615260075536], [0.89048781067281, -0.9448731072649348, 0.0], [0.8890181414440312, 0.0, 0.0]], [[-0.5478352079882176, -0.29712817219553855, -0.9657904855201964], [0.029649730126149354, 0.5975399461436748, 0.0], [-0.5090973639315446, 0.0, 0.0]]]
    Dx, Dy = [[[-0.26186036745135133, -0.06578327856252764, -0.12745939576472742], [0.7569842181651627, -0.2752169357050396, 0.0], [-0.40530836805364623, 0.0, 0.0]], [[-0.8430240372153841, 0.8698443003542284, 0.27207240662186316], [-0.04305933100982973, 0.7230297472936218, 0.0], [0.7693034364225857, 0.0, 0.0]]]
    Ex, Ey = [[[0.9224383103199736, 0.058067176333767456, 0.14228404480935586], [0.7034874711362713, -0.3823675118487939, 0.0], [-0.7253256041678289, 0.0, 0.0]], [[-0.5505891424166067, 0.7465680175138882, 0.619850393088756], [-0.7541783831728857, 0.7276713815242248, 0.0], [0.11807392184738541, 0.0, 0.0]]]
    Fx, Fy = [[[-0.9425805440061719, -0.21926161637198494, 0.12566944651690393], [-0.005605160609513771, -0.23002018174571326, 0.0], [0.9509433634325835, 0.0, 0.0]], [[0.9797012868387667, -0.8211695850285523, 0.06514913178205206], [0.39455062643269745, -0.46728497091024046, 0.0], [-0.6799601231848034, 0.0, 0.0]]]



if __name__ == "__main__":
    # mapping = DoubleMappingFromXY.find_attractor(mag=1, max_degree=2)
    mapping = DoubleMappingFromXY(MappingRecord.Cx, MappingRecord.Cy)

    # mapping1 = DoubleMappingFromXY(MappingRecord.Ax, MappingRecord.Ay)
    # mapping2 = DoubleMappingFromXY(MappingRecord.Fx, MappingRecord.Fy)
    # mapping = RandomChoiceMappingFromXY([mapping1, mapping2], [0.01, 1])
    print(mapping)
    mapping.plot_convergence_points(x_min=-5, x_max=5, y_min=-5, y_max=5, resolution=100, max_iterations=1000, max_abs=10)

    x,y = np.random.uniform(-1, 1, 2)
    scatter_points = get_late_trajectory(mapping, x, y, min_index=1000, max_index=10000)
    if trajectory_diverges(scatter_points):
        print("divergent")
        sys.exit()
    print("number of unique points:", len(set(scatter_points)))
    xs = [p[0] for p in scatter_points]
    ys = [p[1] for p in scatter_points]
    plt.scatter(xs, ys, marker=".", s=2)
    plt.savefig("out.png")
