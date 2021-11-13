import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
from datetime import datetime


class DivergenceError(Exception):
    pass


class Mapping:
    def converges(self):
        try:
            self.find_convergent_initial_condition(min_index=100, max_index=1000)
            return True
        except DivergenceError:
            return False

    def converges_with_complexity(self):
        if not self.converges():
            return False, None, None, None
        min_unique_ratio = 0.1
        n_trials = 10
        total_points = 0
        total_unique_points = 0
        for i in range(n_trials):
            x0, y0, trajectory = self.find_convergent_initial_condition(min_index=100, max_index=1000)
            n_unique_points = len(set(trajectory))
            unique_ratio = n_unique_points / len(trajectory)
            total_points += len(trajectory)
            total_unique_points += n_unique_points
            if unique_ratio >= min_unique_ratio:
                return True, n_unique_points, len(trajectory), unique_ratio
        total_unique_ratio = total_unique_points / total_points
        return False, total_unique_points, total_points, total_unique_ratio

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

    def find_convergent_initial_condition(self, min_index, max_index):
        # print("finding convergent initial condition")
        max_iterations = 100
        counter = 0
        while True:
            counter += 1
            if counter % 100 == 0:
                print(f"now at {counter} iterations in search for convergent initial condition")
            if counter > max_iterations:
                raise DivergenceError(f"could not find convergent initial condition")
            # r = np.log(1 + counter)  # start with close to origin, try going farther out over time
            r = np.random.normal(0, 1)
            theta = np.random.uniform(0, 2*np.pi)
            x0 = r * np.cos(theta)
            y0 = r * np.sin(theta)
            # x0, y0 = np.random.uniform(-1, 1, (2,))
            trajectory = get_late_trajectory(self, x0, y0, min_index=min_index, max_index=max_index, max_abs=1e6)
            if trajectory_diverges(trajectory):
                continue
            else:
                break
        # print("done finding convergent initial condition")
        return x0, y0, trajectory

    def plot_attractor(self):
        # x,y = np.random.uniform(-1, 1, 2)
        x0, y0, short_late_trajectory = self.find_convergent_initial_condition(min_index=1000, max_index=1100)
        scatter_points = get_late_trajectory(self, x0, y0, min_index=1000, max_index=100000)
        # scatter_points = late_trajectory
        if trajectory_diverges(scatter_points):
            print("initial state was thought to converge but late trajectory diverges")
            raise DivergenceError
        print("number of unique points:", len(set(scatter_points)))
        xs = [p[0] for p in scatter_points]
        ys = [p[1] for p in scatter_points]
        plt.scatter(xs, ys, marker=".", s=2)


class SingleMappingFromXY(Mapping):
    # maps a pair of coords to a single new value
    def __init__(self, coefficients):
        coefficients = np.array(coefficients)
        n_x_powers, n_y_powers = coefficients.shape
        self.x_powers, self.y_powers = np.meshgrid(range(n_x_powers), range(n_y_powers))
        self.coefficients = coefficients

    def perturb(self, amount):
        # creates a new mapping where some random vector of length `amount` has been added to the coefficients
        v = np.random.normal(0, 1, self.coefficients.shape)
        v[self.coefficients == 0] = 0  # don't increase the degree
        v *= (amount / np.linalg.norm(v))  # normalize magnitude
        c = v + self.coefficients
        # print(f"perturbed\n{self.coefficients}\nto\n{c}")
        return SingleMappingFromXY(c)

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

    def perturb(self, amount_per_equation):
        # perturbs the x mapping and the y mapping each by adding a vector of length `amount_per_equation`, returns new mapping
        mx = self.mapping_x.perturb(amount_per_equation)
        my = self.mapping_y.perturb(amount_per_equation)
        return DoubleMappingFromXY.from_single_mappings(mx, my)

    @staticmethod
    def from_single_mappings(mapping_x, mapping_y):
        cx = mapping_x.coefficients
        cy = mapping_y.coefficients
        return DoubleMappingFromXY(cx, cy)

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
        print("finding mapping with attractor state")
        while True:
            mapping = DoubleMappingFromXY.random(mag, max_degree)
            x,y = np.random.uniform(-1, 1, 2)
            scatter_points = get_late_trajectory(mapping, x, y, min_index=1000, max_index=10000)
            if trajectory_diverges(scatter_points):
                # print("divergent")
                continue
            n_unique_points = len(set(scatter_points))
            unique_ratio = n_unique_points / len(scatter_points)
            if unique_ratio < 0.1:
                print(f"not enough unique points: {n_unique_points}")
                continue
            print("done finding mapping with attractor state")
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

    def __repr__(self):
        list_of_mapping_reprs = [repr(mapping) for mapping in self.mappings]
        str_of_mapping_reprs = "\n\n\t".join(list_of_mapping_reprs)
        return f"<Random choice mapping from the sub-mappings:\n\t{str_of_mapping_reprs}\n>"


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


def run_perturbation_experiment(perturbation_nelda):
    mapping = DoubleMappingFromXY.find_attractor(mag=1, max_degree=3)
    perturbation = 10 ** (-perturbation_nelda)
    t0 = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    plt.gcf().clear()
    try:
        mapping.plot_attractor()
    except DivergenceError:
        return
    dr = f"StrangeAttractorImages/perturbation/perturbation_{t0}/"
    if not os.path.exists(dr):
        os.mkdir(dr)
    plt.savefig(dr + f"/perturbation_{t0}_nelda{perturbation_nelda}_original.png")
    with open(dr + f"perturbation_{t0}_nelda{perturbation_nelda}_original_coefficients.txt", "w") as f:
        f.write(repr(mapping))

    n_perturbations_to_get = 10
    perturbation_counter = 0
    while perturbation_counter < n_perturbations_to_get:
        new_mapping = mapping.perturb(perturbation)
        if not new_mapping.converges():
            print("new mapping does not converge at all")
        converges_complexly, n_unique_points, n_points, unique_ratio = new_mapping.converges_with_complexity()
        if not converges_complexly:
            print(f"new mapping does not converge complexly: {n_unique_points} unique points out of {n_points} points, for a ratio of {unique_ratio}")
            continue
        else:
            print(f"found complexity in perturbed mapping: {n_unique_points} unique points out of {n_points} points, for a ratio of {unique_ratio}")
        plt.gcf().clear()
        # mapping1.plot_attractor()
        # mapping2.plot_attractor()
        try:
            new_mapping.plot_attractor()
        except DivergenceError:
            continue

        # plt.show()
        # plt.savefig(f"StrangeAttractorImages/K_AS/K_AS_alpha_{alpha:.04f}.png")
        plt.savefig(dr + f"perturbation_{t0}_nelda{perturbation_nelda}_{perturbation_counter}.png")
        with open(dr + f"perturbation_{t0}_nelda{perturbation_nelda}_{perturbation_counter}_coefficients.txt", "w") as f:
            f.write(repr(new_mapping))
        perturbation_counter += 1


class MappingRecord:
    Ax, Ay = [[[-0.61812953, 0.62618444, 0.95847548], [-0.28566655, 0.60419982, 0], [-0.82349836, 0, 0]], [[0.41322135, -0.056517, 0.09507931], [0.84468446, -0.05060947, 0], [-0.9864845, 0, 0]]]  # two fractal leaves
    Bx, By = [[[0.011100313095467884, -0.2653760753248544, 0.7175056845186025], [-0.6633260488831336, 0.16221771794919104, 0.0], [-0.6300085491546239, 0.0, 0.0]], [[0.9796117830195128, -0.8743965549320505, -0.9644515017383146], [0.12238901385759271, 0.3614148282517575, 0.0], [0.3019429543542915, 0.0, 0.0]]]  # sandpaper
    Cx, Cy = [[[-0.4263134430524571, -0.2274150264834227, 0.07281615260075536], [0.89048781067281, -0.9448731072649348, 0.0], [0.8890181414440312, 0.0, 0.0]], [[-0.5478352079882176, -0.29712817219553855, -0.9657904855201964], [0.029649730126149354, 0.5975399461436748, 0.0], [-0.5090973639315446, 0.0, 0.0]]]  # slightly warped circle
    Dx, Dy = [[[-0.26186036745135133, -0.06578327856252764, -0.12745939576472742], [0.7569842181651627, -0.2752169357050396, 0.0], [-0.40530836805364623, 0.0, 0.0]], [[-0.8430240372153841, 0.8698443003542284, 0.27207240662186316], [-0.04305933100982973, 0.7230297472936218, 0.0], [0.7693034364225857, 0.0, 0.0]]]  # pointy hook
    Ex, Ey = [[[0.9224383103199736, 0.058067176333767456, 0.14228404480935586], [0.7034874711362713, -0.3823675118487939, 0.0], [-0.7253256041678289, 0.0, 0.0]], [[-0.5505891424166067, 0.7465680175138882, 0.619850393088756], [-0.7541783831728857, 0.7276713815242248, 0.0], [0.11807392184738541, 0.0, 0.0]]]  # sandpaper
    Fx, Fy = [[[-0.9425805440061719, -0.21926161637198494, 0.12566944651690393], [-0.005605160609513771, -0.23002018174571326, 0.0], [0.9509433634325835, 0.0, 0.0]], [[0.9797012868387667, -0.8211695850285523, 0.06514913178205206], [0.39455062643269745, -0.46728497091024046, 0.0], [-0.6799601231848034, 0.0, 0.0]]]  # peacock tail
    Gx, Gy = [[[-0.1532753711690451, -0.1576121173599503, 0.430583761987954], [-0.8257217296756896, -0.6168485660661247, 0.0], [-0.6795906956107203, 0.0, 0.0]], [[0.4624245068474737, 0.2857576008709981, -0.8495805340760403], [0.9924895335700097, 0.29506081246754134, 0.0], [-0.3240113993589506, 0.0, 0.0]]]  # oval
    Hx, Hy = [[[0.25966105442250886, -0.7939967270586219, 0.42917774892589655], [-0.5440820094127197, 0.1925955328572555, 0.0], [0.9485447921669079, 0.0, 0.0]], [[-0.837298229287887, 0.255105337960418, 0.9037783116305338], [0.26119924941535166, -0.10426171424878139, 0.0], [0.05289386469118562, 0.0, 0.0]]]  # four barbed pentagons
    Ix, Iy = [[[-0.6742848493450075, -0.48068688464786624, 0.8621981389262696], [0.12433753349146048, 0.7788448141920097, 0.0], [0.08463687902844752, 0.0, 0.0]], [[-0.6782791243629129, 0.9818862999296545, -0.7414694201272127], [-0.8045453352013925, -0.47868874451710797, 0.0], [0.13086602112008494, 0.0, 0.0]]]  # tornado
    Jx, Jy = [[[-0.3380695670783209, -0.5388309782183831, 0.09854461708473194], [-0.07316263296971393, 0.9771591776569704, 0.0], [0.4795186207487192, 0.0, 0.0]], [[0.9816564663911391, 0.9374966021288151, -0.8615438028800741], [-0.5795567272340942, -0.9152883631105793, 0.0], [0.7506239670233701, 0.0, 0.0]]]  # dented circle with spines
    Kx, Ky = [[[-0.9034081268946901, -0.41931649294459894, 0.4522682424342557], [-0.1646074980158272, 0.6418454961616755, 0.0], [0.2886990050530647, 0.0, 0.0]], [[-0.28742235875474553, -0.9226778238148496, 0.7098481018173008], [-0.7160331241758036, 0.47781043960632275, 0.0], [0.28701452793694227, 0.0, 0.0]]]  # isolated line segments on arc
    Lx, Ly = [[[0.45433284666595, -0.22433725728001574, 0.13749052353661173], [-0.34337338778816906, -0.821710379916532, 0.0], [0.18713588089388233, 0.0, 0.0]], [[-0.5890842658346485, 0.11007946068208674, 0.5559296907704996], [0.3017419833370627, 0.7847152698358897, 0.0], [0.010059523941953197, 0.0, 0.0]]]  # 7 isolated points on arc
    Mx, My = [[[0.7798852406908356, -0.9997671504400556, -0.7478540352120224], [0.631523405342455, -0.34335272770627734, 0.0], [-0.9500718792027345, 0.0, 0.0]], [[0.006332624057042935, 0.043157071422877724, 0.2858721885486819], [0.06591514388387587, -0.6191818307553068, 0.0], [0.1328227741039496, 0.0, 0.0]]]  # two fractal bulbs
    Nx, Ny = [[[-0.7260651096899733, -0.4021935751787411, 0.012765933742349578], [-0.5211170651794905, 0.05985086443929433, 0.0], [0.5739087084404295, 0.0, 0.0]], [[0.917608505529188, -0.5163301206228947, 0.7104543863226793], [-0.17157610981845628, -0.9499462820489186, 0.0], [-0.7849830844781942, 0.0, 0.0]]]  # rounded triangle fractal
    Ox, Oy = [[[-0.6290903265933283, -0.9651309828805865, -0.6657788091025698], [0.5452314809497985, 0.8055547994436691, 0.0], [0.7057036996439501, 0.0, 0.0]], [[-0.8377501475265969, -0.44729203498641423, 0.3929142342745249], [0.8136478703525969, -0.8461709025182849, 0.0], [0.8326167975997667, 0.0, 0.0]]]  # sharkfin in water
    Px, Py = [[[-0.07428815193849436, -0.6593855685370176, -0.4499824836382418], [-0.6823176250355307, 0.8249475447290009, 0.0], [-0.8353792609832067, 0.0, 0.0]], [[-0.8356261211300378, -0.09290204841049499, 0.6173433532257107], [0.897133993417865, 0.6007860208668772, 0.0], [0.1426634132820277, 0.0, 0.0]]]  # triangle sail
    Qx, Qy = [[[0.7955419908168555, 0.4662918810682737, -0.8808963514773149], [0.6184195112588926, -0.4504639119130487, 0.0], [-0.03808955216886223, 0.0, 0.0]], [[0.7088097422424475, 0.1909064588577205, 0.8472975203311031], [-0.7827173180545954, 0.4118107119006158, 0.0], [-0.05767648539560155, 0.0, 0.0]]]  # two twisted loops
    Rx, Ry = [[[-0.8430312408224176, -0.08877017721341374, 0.8061728813244049], [-0.5358823687986027, -0.08220296534496807, 0.0], [0.12991810888858946, 0.0, 0.0]], [[0.1652747826156098, -0.6673788990356064, -0.3927870826061932], [-0.7514058145725768, 0.45825936967977565, 0.0], [-0.5907618205934937, 0.0, 0.0]]]  # two pointy fish
    Sx, Sy = [[[0.9446700115281086, 0.4298452564544206, -0.2996748369816542], [-0.5363877836280637, 0.2553249440624479, 0.0], [-0.21468204536464652, 0.0, 0.0]], [[0.2520276944342552, -0.8766772052300915, 0.4050380703538712], [-0.915638590243957, -0.9750245529815837, 0.0], [-0.5956174754376151, 0.0, 0.0]]]  # swishy arch
    Tx, Ty = [[[-0.3642761651637316, -0.33348272740761287, 0.0890293026648854], [0.9651449627851774, 0.08221563911903429, 0.0], [-0.5423278748392466, 0.0, 0.0]], [[0.23445158782579578, 0.18312999224860071, 0.18841115738543524], [-0.7812760741235414, -0.2540651010935795, 0.0], [-0.6598230731214718, 0.0, 0.0]]]  # line with central density
    Ux, Uy = [[[0.9013735557304661, -0.24043260775918474, -0.5268379518995354], [-0.6242013964232849, 0.6789896221734257, 0.0], [-0.35026603632497477, 0.0, 0.0]], [[-0.8419252078423902, -0.009814614218918605, 0.25456771157162517], [0.9315046636186068, 0.1981498617551707, 0.0], [0.10635512909550182, 0.0, 0.0]]]  # danish
    Vx, Vy = [[[0.5236592468047314, 0.042083419085794604, -0.4415219169372602], [0.8503512302334728, 0.9978802765276715, 0.0], [-0.62126879366457, 0.0, 0.0]], [[0.8355025207885136, 0.6372465253188384, 0.3393292747462151], [0.5261980017669783, -0.6322689133464883, 0.0], [-0.2720488767868605, 0.0, 0.0]]]  # danish with two central wires
    Wx, Wy = [[[-0.0646027851324773, 0.03807582784311103, -0.22538160643264815], [0.6534227110370547, -0.7928482743219303, 0.0], [0.2170808859463642, 0.0, 0.0]], [[-0.9267081108893946, -0.6407442871876758, 0.9346077142779559], [-0.15941770048968773, -0.9428438332476075, 0.0], [0.10578192014875243, 0.0, 0.0]]]  # spiky loop
    Xx, Xy = [[[0.11302196157575617, 0.5738122932936471, 0.30897588469555504], [-0.23922792640200807, -0.01902496991850322, 0.0], [-0.7894027427905943, 0.0, 0.0]], [[-0.6929027594094999, -0.16798467767699887, 0.5218266740675683], [-0.5212225613708921, -0.14121174736726405, 0.0], [0.513485943870472, 0.0, 0.0]]]  # another line with central density
    Yx, Yy = [[[-0.6702692646955344, -0.8115376317759324, 0.02223417560555152], [0.5308840770353285, 0.08480852973626107, 0.0], [0.10719084397303491, 0.0, 0.0]], [[-0.7974606845699019, -0.2980923118636132, 0.00670460381694804], [0.17905519801152447, 0.811653105688404, 0.0], [0.9864197431187058, 0.0, 0.0]]]  # pentagon-like circle
    Zx, Zy = [[[0.667961136248705, -0.9205966836434327, 0.046558134517790783], [0.15046772873776626, -0.11545291874919283, 0.0], [-0.50859596703938, 0.0, 0.0]], [[-0.3156849726199862, -0.09653731183228609, -0.610518502125331], [-0.9628640750200128, 0.3526305890925485, 0.0], [0.9681984971068758, 0.0, 0.0]]]  # spiky loop country border
    AAx, AAy = [[[-0.7370287960424435, -0.30052836258206517, 0.9236248938031464], [-0.9008298318496966, 0.7319221256195314, 0.0], [0.7789667155543063, 0.0, 0.0]], [[-0.2794214392849774, -0.9617029941536104, -0.09169836349365634], [0.14191101659325334, 0.5391090600243158, 0.0], [0.9045455322199658, 0.0, 0.0]]]  # two spiky feathers
    ABx, ABy = [[[-0.25192598701757274, -0.7496643146393476, 0.27077967021970806], [-0.7638472318950549, 0.2175416349909325, 0.0], [0.5344170631080964, 0.0, 0.0]], [[-0.995370463475657, -0.3236678778917119, 0.4615464396987572], [0.9264269602708433, 0.03388238845901426, 0.0], [0.28917024352955845, 0.0, 0.0]]]  # spiderweb net
    ACx, ACy = [[[-0.4547501939392615, 0.9304490246287591, 0.10256269645710758], [0.899890121537904, -0.16547925694675913, 0.0], [0.2592117800843603, 0.0, 0.0]], [[0.05302751519764293, -0.49326518712369616, 0.6220152886964343], [-0.27667389761474004, -0.8509088825862965, 0.0], [-0.6441639225379017, 0.0, 0.0]]]  # banded complex loop (very cool!)
    ADx, ADy = [[[-0.6791954451383677, -0.8737182616076931, -0.6307493466695118, -0.5802801337437309, 0.5434951645967914], [-0.3488703438788048, 0.5207071104048815, 0.2003371175341937, 0.3578915104634932, 0.0], [0.34790941316488966, -0.13189879211804345, -0.15807615426745913, 0.0, 0.0], [0.4871854697962543, -0.1657996437203637, 0.0, 0.0, 0.0], [0.4494315154169808, 0.0, 0.0, 0.0, 0.0]], [[-0.07424657719764394, 0.9412082874666832, -0.9173280686293432, 0.16856273689041124, 0.8198262530399953], [0.6744106316036569, 0.11530806099995616, 0.9970525516440107, -0.9222708452405861, 0.0], [0.9981169351088437, 0.06812957577404632, 0.1354434015369992, 0.0, 0.0], [0.2374456295731242, 0.8231449244517288, 0.0, 0.0, 0.0], [0.11301827110997698, 0.0, 0.0, 0.0, 0.0]]]  # banded winged triangular loop (quartic degree)
    AEx, AEy = [[[-0.010526797587123049, 0.18206014505438595, -0.22639721411622316, -0.7858882948394639, 0.9516077950078414], [0.354522021100923, -0.6530385187822338, 0.3162862529176902, 0.19392799948054007, 0.0], [-0.9262519936700027, -0.7648577892684996, -0.28705586455625887, 0.0, 0.0], [-0.2619068947735288, 0.46879138200299497, 0.0, 0.0, 0.0], [0.058531498337911714, 0.0, 0.0, 0.0, 0.0]], [[-0.7744357812517002, 0.15129628426589314, 0.823478185056195, -0.10325255629786079, 0.6868399677485362], [0.3496068835321555, 0.4808055768712922, 0.25441106999710494, -0.1578642505967638, 0.0], [-0.2686623828266512, -0.7897400877791643, 0.7749253368838216, 0.0, 0.0], [-0.22223717820174627, -0.1450931499769872, 0.0, 0.0, 0.0], [-0.09791631968513159, 0.0, 0.0, 0.0, 0.0]]]  # loop with three internal teeth (quartic degree)
    AFx, AFy = [[[-0.14754097612270134, 0.3454997155090238], [0.429427999011335, 0.0]], [[-0.12416788830932979, 0.5749391471612384], [-0.8120235520578423, 0.0]]]  # line (linear degree)
    AGx, AGy = [[[-0.7750562873904929, -0.6180111286243533], [0.5967851740650081, 0.0]], [[-0.10189219415296935, 0.4127385484006276], [0.8364772737101478, 0.0]]]  # comet (linear degree)
    AHx, AHy = [[[-0.27211562844484916, -0.8319716689852976, 0.9351408235409293], [-0.6823195045946777, -0.6466249822249379, 0.0], [0.21965031740814345, 0.0, 0.0]], [[0.3921564315754067, 0.08856161846112665, -0.9482557710280228], [-0.6755528928100367, -0.4717489420532923, 0.0], [-0.07280200384859548, 0.0, 0.0]]]  # spiral with central density
    AIx, AIy = [[[-0.8393957755264603, -0.8879044965874197, 0.4230348570284028], [0.6149522860773156, 0.1264888905244701, 0.0], [0.8454462599804577, 0.0, 0.0]], [[-0.7141775215624688, -0.02903871845789663, 0.6336930666378235], [0.8696213045771932, -0.05841679454473758, 0.0], [0.14384061629159173, 0.0, 0.0]]]  # two teardrops which are mostly very diffuse, except for a set of dense line segments
    AJx, AJy = [[[-0.7581566467104306, 0.28078444498513555, 0.7903677663262791], [-0.7241253174456495, 0.9498787797471524, 0.0], [0.4004652769246815, 0.0, 0.0]], [[0.9012269177117744, -0.8070384262286234, -0.18318698991741122], [0.21849448737745858, 0.0409457754819853, 0.0], [-0.5270238583914626, 0.0, 0.0]]]  # two complex triangles warped into an arc
    AKx, AKy = [[[0.6807006641027684, -0.23193500162667657, -0.1320542078667737], [-0.0013246833211328912, -0.39007247219561636, 0.0], [0.6257671921844892, 0.0, 0.0]], [[0.52800490183569, 0.17147809364256705, -0.46969642327009775], [0.662708356926657, -0.2435696461563266, 0.0], [0.43558001748091124, 0.0, 0.0]]]  # oval
    ALx, ALy = [[[0.5099057895906778, -0.8677057210489232, 0.34433607635655594], [-0.732111323801578, 0.6610541070638256, 0.0], [-0.9566236476137928, 0.0, 0.0]], [[0.4358787208215771, 0.2340757691250117, 0.7451496902435053], [-0.516209679730697, -0.42741690331006366, 0.0], [-0.7549676646021979, 0.0, 0.0]]]  # five-armed spiral with central density
    AMx, AMy = [[[0.5499220954832924, 0.9296521752256588, -0.8684267855800054], [0.9905741011796014, -0.5535707136679064, 0.0], [-0.46453033236428976, 0.0, 0.0]], [[-0.4152467111389291, 0.2043127601739947, -0.0680040521561236], [-0.8578424048131852, -0.21686756999112977, 0.0], [-0.045749501983700425, 0.0, 0.0]]]  # two loops, like teardrops, whole thing has rot2 symmetry
    ANx, ANy = [[[-0.22991893507646988, -0.9219582762557677, 0.4019043169620571], [0.332410589933005, -0.24712759463241696, 0.0], [0.3910688517890635, 0.0, 0.0]], [[-0.25779213579646343, 0.5528870612053325, -0.8218199417021359], [-0.4513433822118249, -0.8122499581085585, 0.0], [0.42258955659519115, 0.0, 0.0]]]  # two hooks
    AOx, AOy = [[[-0.9318067697171788, -0.5929223864177999, -0.15028007054071413], [-0.48855398627195923, 0.4004506584919276, 0.0], [0.9175683379257229, 0.0, 0.0]], [[-0.20952951688167754, 0.49507986632936185, -0.12648652544968408], [-0.9895401249019649, 0.05884124209496622, 0.0], [0.6041892754126164, 0.0, 0.0]]]  # banded pentagonal loop
    APx, APy = [[[0.846916547480272, 0.9302074473284354, -0.4621234138886654], [-0.8156023863486168, 0.10192063825521269, 0.0], [-0.7322791925910592, 0.0, 0.0]], [[0.3421648923132392, -0.14452501314808197, -0.7711593150115379], [-0.20428039728845304, 0.5254086784159349, 0.0], [0.6930038918422219, 0.0, 0.0]]]  # thick triangle
    AQx, AQy = [[[0.21978916885170374, -0.5404118542799088, 0.600103574920656], [-0.5550526949929011, 0.40794246691380187, 0.0], [-0.3723281380279855, 0.0, 0.0]], [[0.6289274598585626, 0.40801534506961934, -0.9850829990492258], [-0.9328029430238862, -0.9350941864697262, 0.0], [-0.31979540142757945, 0.0, 0.0]]]  # complex curvy leaf
    ARx, ARy = [[[0.8117940962119976, -0.4649971127921413], [-0.8887178003196341, 0.0]], [[0.3258012373210373, -0.742761249782802], [0.2355584065539742, 0.0]]]  # line seemingly without density gradient (linear degree)
    ASx, ASy = [[[-0.07643804233090146, -0.15680941623093836], [0.16742021870347545, 0.0]], [[0.4127171473490223, 0.5271568655076095], [-0.886289791719957, 0.0]]]  # line with central density (linear degree)
    ATx, ATy = [[[-0.9469038431167203, 0.5146927809484609, -0.24388414668947567], [0.16532112389818376, 0.683313502786961, 0.0], [0.05464388876518478, 0.0, 0.0]], [[-0.7867237682661723, -0.588321444771319, -0.4446193487948711], [-0.8684343868626598, 0.4368623082912675, 0.0], [0.19270797193410028, 0.0, 0.0]]]  # danish
    AUx, AUy = [[[-0.0448042088488132, -0.026871250923233836, 0.9071538175585814, 0.1519682708941663, 0.1735099446788788], [0.6148565523413956, 0.5783766092817089, -0.9716609029061594, -0.14550563690820106, 0.0], [0.2677035731578039, -0.3017079617155052, 0.9189759536575746, 0.0, 0.0], [-0.015898339562792962, -0.17257202425325957, 0.0, 0.0, 0.0], [-0.4400286662521089, 0.0, 0.0, 0.0, 0.0]], [[-0.26801748639454703, 0.8273046034970095, -0.8638963680512879, -0.046828396629829516, 0.18952486552111436], [0.8015148590510217, -0.32254949634797514, -0.9735339271574006, 0.041567126704104895, 0.0], [0.9015119565098915, -0.4161951052512569, -0.1278128981912603, 0.0, 0.0], [-0.9689077987671502, -0.16299725884624294, 0.0, 0.0, 0.0], [0.98108090406905, 0.0, 0.0, 0.0, 0.0]]]  # six leaf loops (quartic degree)
    AVx, AVy = [[[0.6392670719401352, -0.6586485432949705, -0.3607584382687343, -0.42345237429443916, -0.9902525500369281], [-0.13311354777889806, 0.27219399938959943, -0.359371346592952, 0.30339700923418556, 0.0], [-0.3809801486771769, 0.44582057673888964, 0.8522732970002267, 0.0, 0.0], [-0.42937089999471745, 0.6234130651514471, 0.0, 0.0, 0.0], [-0.19176960675060872, 0.0, 0.0, 0.0, 0.0]], [[-0.6404131020005623, 0.5943806532788953, 0.5700184030116144, -0.47423886921886793, 0.47308824840772656], [0.7934776502463567, 0.8880918648865861, -0.39267198565536177, -0.44318452765967153, 0.0], [0.8164919036049085, 0.5911455469637585, 0.8477145170110525, 0.0, 0.0], [-0.7015081916325443, 0.9915632345409082, 0.0, 0.0, 0.0], [0.6486815235822567, 0.0, 0.0, 0.0, 0.0]]]  # twisted leaves (quartic degree)
    AWx, AWy = [[[-0.5180704777548775, 0.9528127112653579, 0.5346248524158692, -0.3527398325297635], [-0.9497139720445742, 0.29992814350146335, 0.29990808237609956, 0.0], [-0.8435693452112623, -0.47199278707760883, 0.0, 0.0], [0.6431163467287095, 0.0, 0.0, 0.0]], [[-0.5247607820751827, -0.3813735187716809, -0.173873116714649, -0.3198480970929247], [-0.16859427773075297, 0.3684896155625157, -0.6578919453144063, 0.0], [-0.8529510790745869, -0.05648661051223702, 0.0, 0.0], [0.6009191673287364, 0.0, 0.0, 0.0]]]  # sword blade (cubic degree)
    AXx, AXy = [[[-0.38258246438483834, -0.005201017951926401, 0.17460134128766902, 0.36215857392309636], [-0.8180205080879146, 0.5462880612849719, 0.036187513445569364, 0.0], [-0.9613810854582976, 0.482106254098976, 0.0, 0.0], [0.4973527539700302, 0.0, 0.0, 0.0]], [[0.4014306295233614, 0.926485378483493, -0.8817942565807413, 0.34460135027614003], [-0.023058243688384827, 0.25577920002851595, 0.5034587499440282, 0.0], [-0.6059073128323387, -0.0052300343351059375, 0.0, 0.0], [0.0315536551734541, 0.0, 0.0, 0.0]]]  # X made of hooked leaves (cubic degree)
    AYx, AYy = [[[0.03286393687280542, -0.5741864544042616, 0.6254522327689518, 0.9281403263701193], [0.496854598035948, -0.6011007672959343, -0.6330300245837586, 0.0], [0.1354806554720147, 0.4759175357801715, 0.0, 0.0], [-0.37179831412353725, 0.0, 0.0, 0.0]], [[-0.9394966943933196, -0.1838571570024976, -0.5667303152102932, 0.4781021993691146], [0.6395281072250851, -0.33787490060477254, -0.9728327341957506, 0.0], [0.8467384136489782, -0.7469946478023062, 0.0, 0.0], [-0.8113143871867603, 0.0, 0.0, 0.0]]]  # two swoops (cubic degree)
    AZx, AZy = [[[0.6509367927504683, -0.08752292700961095, 0.8986021008562286, -0.6150952016914468], [-0.30440055897944385, 0.6736570706309368, 0.1596057884372386, 0.0], [-0.3036443078715356, -0.43376560144215204, 0.0, 0.0], [0.2661433021451334, 0.0, 0.0, 0.0]], [[-0.10952214707285268, -0.6409615134266577, 0.5340933913070518, -0.9004496277781597], [-0.7123875567473834, 0.8570406175085687, 0.9849731279307556, 0.0], [-0.23167136462777793, 0.3120660503396455, 0.0, 0.0], [0.6475146087944248, 0.0, 0.0, 0.0]]]  # pentagonal banded twist (cubic degree)
    BAx, BAy = [[[0.009328904575528485, 0.6939951400524438, -0.3092213851308656, 0.6525483835883261], [-0.2303070950968782, -0.7944876544199464, -0.4494308662252635, 0.0], [-0.5891744233783633, -0.5431480955072889, 0.0, 0.0], [0.9090235467436292, 0.0, 0.0, 0.0]], [[0.8670662556374398, -0.5488648605611777, 0.2414696514114565, 0.39641981695244155], [0.43336386092968815, 0.4503379280648341, -0.2232709867322158, 0.0], [0.5814336946364582, -0.7002933008677303, 0.0, 0.0], [-0.8472148630503407, 0.0, 0.0, 0.0]]]  # starfish tornado (cubic degree)
    BBx, BBy = [[[0.4402237884355531, 0.2937074048881718, -0.632418145306394, -0.8333300375749026], [-0.315063651330624, -0.9385815615842463, 0.2609496141116854, 0.0], [-0.7975024810391911, 0.10862626944606624, 0.0, 0.0], [0.30931511981429227, 0.0, 0.0, 0.0]], [[0.6486532020275668, -0.4928487014975642, -0.34341581646433283, -0.8975409028481292], [0.7745244100391402, -0.9851966252759037, -0.18440096020624197, 0.0], [-0.704714174365181, -0.3770375273985247, 0.0, 0.0], [-0.28082385750817496, 0.0, 0.0, 0.0]]]  # seashell, loops interrupted by line / sharp turns (cubic)
    BCx, BCy = [[[0.17972191993518782, 0.3629812127205372, -0.06317376697026655, 0.025587529147727217], [0.7242286029568823, -0.5839655183016759, -0.5164409394865392, 0.0], [-0.5156693395716214, 0.8299521937979639, 0.0, 0.0], [0.07410480061709412, 0.0, 0.0, 0.0]], [[0.4485691804219678, 0.18797465912560773, 0.9518732561311154, 0.9402787490531597], [0.6649170342849917, 0.10530080334097525, 0.778324904456783, 0.0], [-0.8782127165101417, -0.8379989615466286, 0.0, 0.0], [-0.026399526066826384, 0.0, 0.0, 0.0]]]  # seashell loops with sharp hairpins/thorns (cubic)
    BDx, BDy = [[[-0.7684975294734846, 0.7637040512521869, 0.05300970901529345, -0.42318581576930914], [-0.2881704225279642, 0.35725340052136945, 0.6840266448524375, 0.0], [0.40689589600284326, -0.7305189342468226, 0.0, 0.0], [-0.6464574815235757, 0.0, 0.0, 0.0]], [[-0.8997207375395717, -0.2222097487387964, 0.9114713953800149, -0.17176512497512597], [0.5825462234069643, -0.5038904719035555, -0.09041069192766504, 0.0], [0.45141340954064835, -0.15394322084201972, 0.0, 0.0], [-0.8833775618981721, 0.0, 0.0, 0.0]]]  # parabolic spikes (cubic)
    BEx, BEy = [[[-0.16493041362447158, -0.9041437947540374, 0.9113901166804084, -0.7263153058229803], [0.6431611624734688, 0.5412627232245006, -0.5092959454792021, 0.0], [-0.24454845252895274, -0.7171049543074732, 0.0, 0.0], [0.18324193022324553, 0.0, 0.0, 0.0]], [[0.6703987942999172, -0.8795599010871593, 0.21897321633423195, -0.24604684388052944], [0.6797342004157245, -0.6259943567828061, -0.7017815851605937, 0.0], [0.7988103357080041, -0.5091836042184552, 0.0, 0.0], [-0.16035500872497233, 0.0, 0.0, 0.0]]]  # cigar with hanging ribbon (cubic)
    BFx, BFy = [[[-0.12018134347427445, 0.29561697018915223, 0.5558945190514257, 0.4990644691893791], [0.3589300752123965, 0.14201968219459316, -0.15902467760498284, 0.0], [-0.21002463382013903, -0.7186913681227309, 0.0, 0.0], [-0.7834464302040056, 0.0, 0.0, 0.0]], [[-0.3902562872178339, 0.923800243780607, -0.7226033530927982, -0.6479784422001502], [0.7645197408577278, -0.8252373698291118, -0.5677297080090389, 0.0], [0.45838673800711716, 0.8969278723362777, 0.0, 0.0], [-0.6895295136665094, 0.0, 0.0, 0.0]]]  # constellation of isolated constellations of constellations (cubic)
    BGx, BGy = [[[-0.9044811178602636, 0.5517951232741856, 0.6751852982247533, -0.09325558741644957], [0.24139112448633315, -0.13201375255277048, 0.3722649160675142, 0.0], [-0.7183621105866616, -0.4062741804946277, 0.0, 0.0], [-0.6144558454092746, 0.0, 0.0, 0.0]], [[0.6469039610236014, 0.7213291185161108, -0.7397902462403094, -0.637353210005617], [0.1563581947208208, 0.9905772589757742, -0.5275224473527791, 0.0], [-0.3911335564248546, 0.8015079494516717, 0.0, 0.0], [0.6123026450671776, 0.0, 0.0, 0.0]]]  # swoopy spiky leaf



    records = [
        [Ax, Ay], [Bx, By], [Cx, Cy], [Dx, Dy], [Ex, Ey], [Fx, Fy], [Gx, Gy], [Hx, Hy], [Ix, Iy], [Jx, Jy], 
        [Kx, Ky], [Lx, Ly], [Mx, My], [Nx, Ny], [Ox, Oy], [Px, Py], [Qx, Qy], [Rx, Ry], [Sx, Sy], [Tx, Ty],
        [Ux, Uy], [Vx, Vy], [Wx, Wy], [Xx, Xy], [Yx, Yy], [Zx, Zy], [AAx, AAy], [ABx, ABy], [ACx, ACy], [ADx, ADy],
        [AEx, AEy], [AFx, AFy], [AGx, AGy], [AHx, AHy], [AIx, AIy], [AJx, AJy], [AKx, AKy], [ALx, ALy], [AMx, AMy], [ANx, ANy],
        [AOx, AOy], [APx, APy], [AQx, AQy], [ARx, ARy], [ASx, ASy], [ATx, ATy], [AUx, AUy], [AVx, AVy], [AWx, AWy], [AXx, AXy],
        [AYx, AYy], [AZx, AZy],
    ]



if __name__ == "__main__":
    # mapping1 = DoubleMappingFromXY.find_attractor(mag=1, max_degree=2)
    # print(f"found mapping: {mapping1}")
    # mapping2 = DoubleMappingFromXY.find_attractor(mag=1, max_degree=2)
    # print(f"found mapping: {mapping2}")

    # mapping1 = DoubleMappingFromXY(MappingRecord.Kx, MappingRecord.Ky)
    # mapping2 = DoubleMappingFromXY(MappingRecord.ASx, MappingRecord.ASy)
    # mapping1 = DoubleMappingFromXY.find_attractor(mag=1, max_degree=2)
    # mapping2 = DoubleMappingFromXY.find_attractor(mag=1, max_degree=2)
    # mapping1 = (lambda record: DoubleMappingFromXY(record[0], record[1]))(random.choice(MappingRecord.records))
    # mapping2 = (lambda record: DoubleMappingFromXY(record[0], record[1]))(random.choice(MappingRecord.records))

    alpha = 0.5
    # mapping = RandomChoiceMappingFromXY([mapping1, mapping2], [alpha, 1-alpha])
    # mapping = DoubleMappingFromXY.find_attractor(mag=1, max_degree=3)
    # mapping = DoubleMappingFromXY(MappingRecord.AZx, MappingRecord.AZy)
    # print(mapping)
    # mapping.plot_convergence_points(x_min=-5, x_max=5, y_min=-5, y_max=5, resolution=100, max_iterations=1000, max_abs=10)

    # run_mixing_experiment()
    while True:
        run_perturbation_experiment(perturbation_nelda=random.choice([1,2,3]))


    # found pairings:
    # - C+T
    # - T+X (two lines combine to make something very much NOT a line!)
    # - X+Y (line plus pentagon-like circle creates sharp pentagon with flyaway loops)
    # - AF + AG (two lines from linear mappings create starburst-looking pile of sticks; surprisingly not also just a line despite being a combination of linear mappings, and, I'd think, therefore also a linear mapping (but the random choice messes with this; they're not actually composed))
    # - AF + AK (line plus oval creates starburst with curly squid arms)
    # - AR + AS (two linear-degree lines create very interesting palm-leaf pattern)
    # - AL + AS (line plus spiral creates pentagon made of scratch marks)
    # - K + AS (isolated segments of arc plus centrally dense line creates multiple arcs/lines intersecting)


