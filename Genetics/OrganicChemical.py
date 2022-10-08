import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import random


class OrganicChemical:
    properties = {
        "leftness": "N", "rightness": "N", "horizontality": "Z",
        "downness": "N", "upness": "N", "verticality": "Z",
        "energy": "R", "size": "R", "density": "R",
    }

    # names are defined by path, some paths may have identical properties
    names = {
        "O": "orinan",
        "R": "R-cretinan",
        "L": "L-cretinan",
        "U": "U-cretinan",
        "D": "D-cretinan",
    }

    def __init__(self, path, 
            leftness, rightness, horizontality,
            downness, upness, verticality,
            energy, size, density):
        self.path = path
        self.path_tuple = to_tuple(path)
        self.path_str = self.get_path_str()
        self.leftness = leftness
        self.rightness = rightness
        self.horizontality = horizontality
        self.downness = downness
        self.upness = upness
        self.verticality = verticality
        self.energy = energy
        self.size = size
        self.density = density

        self.properties = self.get_properties()
        self.name = self.get_name()
        self.designation = self.get_designation()

    @staticmethod
    def from_path(path):
        path = np.array(path)
        xs = path[:, 0]
        ys = path[:, 1]
        distances = (lambda x,y: (x**2+y**2)**0.5)(*path.T)
        energy = np.mean(distances)
        size = max(distances)
        density = energy / size if size != 0 else 0
        leftness = min(xs) * -1
        rightness = max(xs)
        horizontality = rightness - leftness
        downness = min(ys) * -1
        upness = max(ys)
        verticality = upness - downness

        return OrganicChemical(path=path,
            leftness=leftness, rightness=rightness, horizontality=horizontality,
            downness=downness, upness=upness, verticality=verticality,
            energy=energy, size=size, density=density,
        )

    @staticmethod
    def from_path_str(s):
        if s == "O":
            path = np.array([[0,0]])
        else:
            arr = [[0,0]]
            for c in s:
                dx, dy = OrganicChemical.get_dx_dy_from_char(c)
                x, y = arr[-1]
                arr.append([x+dx, y+dy])
            path = np.array(arr)
        return OrganicChemical.from_path(path)

    @staticmethod
    def get_dx_dy_from_char(c):
        return {
            "R": (1, 0), "L": (-1, 0),
            "U": (0, 1), "D": (0, -1),
        }[c]

    @staticmethod
    def get_char_from_dx_dy(dx, dy):
        assert abs(dx) + abs(dy) == 1, f"bad dx and dy: {dx}, {dy}"
        if dx == 1:
            c = "R"
        elif dx == -1:
            c = "L"
        elif dy == 1:
            c = "U"
        elif dy == -1:
            c = "D"
        else:
            raise ValueError
        return c

    def get_path_str(self):
        s = ""
        x0, y0 = 0, 0
        for x1, y1 in self.path[1:]:
            dx = x1-x0
            dy = y1-y0
            c = OrganicChemical.get_char_from_dx_dy(dx, dy)
            s += c
            x0, y0 = x1, y1
        if s == "":
            return "O"  # for orinan
        return s

    def plot(self):
        xs = self.path[:, 0]
        ys = self.path[:, 1]
        plt.plot(xs, ys)
        plt.scatter(xs, ys, c="0.5")
        plt.scatter(xs[-1], ys[-1], c="k")  # endpoint should show unless it's the origin
        plt.scatter(xs[0], ys[0], c="r")  # origin should always show
        plt.title(self.designation)
        plt.gca().set_aspect("equal")

        # pad the plots to all be square
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        x_range = max_x - min_x
        y_range = max_y - min_y
        if x_range < y_range:
            # pad x-range
            padding = y_range * 0.05
            x_center = (min_x + max_x)/2
            xlim = (x_center - y_range/2 - padding, x_center + y_range/2 + padding)
            ylim = (min_y - padding, max_y + padding)
        elif x_range > y_range:
            # pad y-range
            padding = x_range * 0.05
            y_center = (min_y + max_y)/2
            xlim = (min_x - padding, max_x + padding)
            ylim = (y_center - x_range/2 - padding, y_center + x_range/2 + padding)
        else:
            padding = x_range * 0.05
            xlim = (min_x - padding, max_x + padding)
            ylim = (min_y - padding, max_y + padding)

        plt.gca().set_xlim(xlim)
        plt.gca().set_ylim(ylim)

        plt.savefig(f"ChemicalDiagrams/{self.designation}.png")
        plt.gcf().clear()

    def __hash__(self):
        return hash(self.path_str)

    def __eq__(self, other):
        if type(other) is not OrganicChemical:
            return NotImplemented
        return self.path_str == other.path_str

    def get_properties(self):
        res = []
        for k in OrganicChemical.properties:
            x = getattr(self, k)
            res.append(x)
        return tuple(res)

    def get_name(self):
        return OrganicChemical.names.get(self.path_str)

    def has_name(self):
        return self.name is not None

    def get_designation(self):
        return f"{self.name}:{self.path_str}" if self.has_name() else self.path_str

    def to_str(self):
        s = "["
        items = []
        for k, domain in OrganicChemical.properties.items():
            x = getattr(self, k)
            if domain == "N":
                # non-negative int
                item = str(x)
            elif domain == "Z":
                # signed int
                item = ("+" if x > 0 else "\u2218" if x == 0 else "") + str(x)
            elif domain == "R":
                item = f"{x:.4f}"
            else:
                raise ValueError(f"unknown domain {domain}")
            items.append(item)
        s += ", ".join(items)
        s += " (" + self.path_str
        if self.has_name():
            s += " : " + self.name
        s += ")"
        s += "]"
        return s

    def __repr__(self):
        return self.to_str()

    def react_with(self, other):
        path0 = self.path_str if self.path_str != "O" else ""
        path1 = other.path_str if other.path_str != "O" else ""
        # operation should be commutative, create some new path based on the reactants' paths
        # overlay the paths in order from origin (zip)
        # whatever tail is left over from the longer chemical will just be tacked on
        len0 = len(path0)
        len1 = len(path1)
        m = min(len0, len1)
        n = max(len0, len1)
        if m == 0 and n == 0:
            s = "O"
        else:
            s = ""
            for i in range(m):
                c0 = path0[i]
                c1 = path1[i]
                c = OrganicChemical.react_chars(c0, c1)
                if c != "O":  # can happen if bases annihilate
                    s += c
            if n > m:
                tail_path = path0 if len0 == n else path1
                for i in range(m, n):
                    s += tail_path[i]
        return OrganicChemical.from_path_str(s)

    @staticmethod
    def react_chars(c0, c1):
        d = {"R": 0, "U": 90, "L": 180, "D": 270}
        a0 = d[c0]
        a1 = d[c1]
        diff = abs(a0 - a1)

        # if opposites, cancel
        if diff == 180:
            return "O"

        # if same, same
        elif diff == 0:
            return c0

        # otherwise, they are at 90 degrees, add the next one around the circle
        a0, a1 = (a0, a1) if a0 < a1 else (a1, a0)
        return {
            (0, 90): "L",
            (90, 180): "D",
            (180, 270): "R",
            (0, 270): "U",
        }[(a0, a1)]

    @staticmethod
    def get_charge_repulsion(x, y):
        # positive for same sign
        sx = 1 if x > 0 else -1 if x < 0 else 0
        sy = 1 if y > 0 else -1 if y < 0 else 0
        s = sx * sy
        return s * abs(x) * abs(y)

    def get_reaction_properties(self, other):
        product = self.react_with(other)
        h0 = self.horizontality
        h1 = other.horizontality
        h = product.horizontality
        v0 = self.verticality
        v1 = other.verticality
        v = product.verticality

        # need x of the first reactant and y of the second, to preserve horizontal and vertical charges
        x_numer = h1 * v - h * v1
        x_denom = h1 * v0 - h0 * v1
        y_numer = h0 * v - h * v0
        y_denom = h0 * v1 - h1 * v0
        x = x_numer / x_denom if x_denom != 0 else np.nan
        y = y_numer / y_denom if y_denom != 0 else np.nan

        # sometimes x and y won't be positive, in which case the reaction can't happen
        possible = not(x <= 0 or y <= 0 or np.isnan(x) or np.isnan(y))
        if possible:
            # stoichiometry to find amount of reactants and products
            lcm = (lambda a, b: abs(a*b) // math.gcd(a, b))(x_denom, y_denom)
            x_numer_mult = lcm/x_denom
            y_numer_mult = lcm/y_denom
            xt = x_numer * x_numer_mult
            yt = y_numer * y_numer_mult
            assert xt % 1 == 0, xt
            assert yt % 1 == 0, yt
            xt = int(xt)
            yt = int(yt)
            zt = lcm
            gcd = math.gcd(math.gcd(xt, yt), zt)
            xt //= gcd
            yt //= gcd
            zt //= gcd
        else:
            xt, yt, zt = 0, 0, 0

        horizontality_diff = zt*h - (xt*h0 + yt*h1)
        verticality_diff = zt*v - (xt*v0 + yt*v1)
        energy_diff = zt*product.energy - xt*self.energy - yt*other.energy
        size_diff = zt*product.size - xt*self.size - yt*other.size

        # energy diff negative means there is less energy in the product than there was in the reactants, so some was released (exothermic)
        # energy diff positive = endothermic

        # it will take activation energy to overcome repulsion
        # activation energy (spontaneity) not same as endo/exothermicity
        h_repulsion = OrganicChemical.get_charge_repulsion(xt*h0, yt*h1)
        v_repulsion = OrganicChemical.get_charge_repulsion(xt*v0, yt*v1)
        repulsion = h_repulsion + v_repulsion

        if possible:
            behavior = ("easy" if repulsion < 0 else "hard" if repulsion > 0 else "indifferent") + "-" + ("hot" if energy_diff < 0 else "cold" if energy_diff > 0 else "ambient")
        else:
            behavior = "none"

        if possible:
            assert horizontality_diff == 0, "stoichiometry error"
            assert verticality_diff == 0, "stoichiometry error"

        return {
            "first reactant": self.path_str,
            "second reactant": other.path_str,
            "possible": possible,
            "behavior": behavior,
            "repulsion": repulsion,
            "energy_diff": energy_diff,
            "size_diff": size_diff,
            "amount of first reactant": xt,
            "amount of second reactant": yt,
            "amount of product": zt,
        }

    @staticmethod
    def get_all_path_strs_in_order():
        for c in "ORULD":
            yield c
        # take every path from previous iteration (except O) and add the chars to it in order
        for path in OrganicChemical.get_all_path_strs_in_order():
            if path == "O":
                continue
            for c in "RULD":
                yield path + c



def to_tuple(x):
    # https://stackoverflow.com/a/1952655/7376935
    try:
        iterator = iter(x)
    except TypeError:
        # not iterable
        return x
    else:
        # iterable
        return tuple(to_tuple(y) for y in x)


def plot_reaction_types(chems):
    n_chems = len(chems)
    repulsion_arr = np.zeros((n_chems, n_chems))
    energy_diff_arr = np.zeros((n_chems, n_chems))
    for i in range(n_chems):
        for j in range(i, n_chems):
            # DO look at how chemical i interacts with itself
            reaction = chems[i].get_reaction_properties(chems[j])
            repulsion = reaction["repulsion"]
            energy_diff = reaction["energy_diff"]
            repulsion_arr[i][j] = repulsion
            repulsion_arr[j][i] = repulsion
            energy_diff_arr[i][j] = energy_diff
            energy_diff_arr[j][i] = energy_diff

    max_repulsion = abs(repulsion_arr).max()
    max_energy_diff = abs(energy_diff_arr).max()
    # get these vmin and vmax BEFORE setting nan, that causes bugs in getting max abs

    reaction_type_arr = np.zeros(repulsion_arr.shape)
    reaction_type_arr[repulsion_arr == 0] += 10
    reaction_type_arr[repulsion_arr > 0] += 20
    reaction_type_arr[energy_diff_arr == 0] += 1
    reaction_type_arr[energy_diff_arr > 0] += 2
    assert set(np.unique(reaction_type_arr)) - {0,1,2,10,11,12,20,21,22} == set()

    repulsion_arr[repulsion_arr == 0] = np.nan
    energy_diff_arr[energy_diff_arr == 0] = np.nan

    colors_positive = plt.cm.winter(np.linspace(0, 1, 128))[::-1]
    colors_negative = plt.cm.autumn(np.linspace(0, 1, 128))
    colors = np.vstack((colors_negative, colors_positive))
    cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)
    # cmap = plt.get_cmap("RdYlBu").copy()

    # energy_diff_cmap_pos = plt.cm.
    # energy_diff_cmap_neg = plt.cm.autumn(np.linspace(-max_energy_diff, 0, 128))
    # colors2 = plt.cm.gist_heat_r(np.linspace(0, 1, 128))
    # colors = np.vstack((colors1, colors2))
    cmap.set_bad(color="black")

    # discrete colormap for the reaction types, where signs are treated as 0 (negative), 1 (zero), 2 (positive) for dimensions R, ED
    # so values are 0, 1, 2, 10, 11, 12, 20, 21, 22
    # https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
    discrete_cmap = mcolors.ListedColormap([
        "red", "0.25", "purple",  # 0=easy-hot, 1=easy-ambient, 2=easy-cold
        "orange", "0.5", "blue",  # 10=indifferent-hot, 11=indifferent-ambient, 12=indifferent-cold
        "yellow", "0.75", "green",  # 20=hard-hot, 21=hard-ambient, 22=hard-cold
    ])
    bounds = [-0.5, 0.5, 1.5, 2.5, 10.5, 11.5, 12.5, 20.5, 21.5, 22.5]
    norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)

    ax1 = plt.subplot(1,3,1)
    plt.imshow(repulsion_arr, cmap=cmap, vmin=-max_repulsion, vmax=max_repulsion)
    plt.colorbar()
    plt.title("repulsion (-easy +hard)")
    plt.subplot(1,3,2, sharex=ax1, sharey=ax1)
    plt.imshow(energy_diff_arr, cmap=cmap, vmin=-max_energy_diff, vmax=max_energy_diff)
    plt.colorbar()
    plt.title("energy diff (-hot +cold)")
    plt.subplot(1,3,3, sharex=ax1, sharey=ax1)
    plt.imshow(reaction_type_arr, cmap=discrete_cmap, norm=norm)
    plt.colorbar()
    plt.title("reaction type")
    plt.show()


def react_many_chemicals(chems, amounts, temperature):
    # you have a box of various chemicals in different amounts
    # certain pairs will react spontaneously and others require activation energy
    # could try something like just let them bump into each other at random and see if they want to react?
    # need some idea of temperature (affected by exo/endothermic reactions that occur) which contributes to activation energy
    # probabilistic reactions? e.g. if it's a spontaneous reaction, the chance of occurring is higher for very negative repulsion values
    # if reaction requires activation energy, how to incorporate that into the probability of reacting?
    # and what about when the reaction requires a lot of the reactants? I guess it has to happen in those quantized amounts?

    amounts_over_time = {c: [amounts[c]] for c in chems}
    temperature_over_time = [temperature]

    t = 0  # t=0 is when we have initial condition, so after the first reaction it should be t=1
    while t < 100000:
        if t % 1000 == 0:
            print(f"t = {t}")

        chems, amounts, temperature = react_one_pair_from_many_chemicals(chems, amounts, temperature)
        t += 1

        # look at the new amounts, any new chems must get zeros before this
        # current time series length is t-1, and we will append the t-th item
        all_chems = set(amounts_over_time.keys()) | set(amounts.keys())
        for c in all_chems:
            if c not in amounts_over_time:
                # new chemical, so we give it a history of zeros before this
                amounts_over_time[c] = [0] * (t - 1)
            amount = amounts.get(c, 0)
            amounts_over_time[c].append(amount)

        temperature_over_time.append(temperature)

    top_chems = [c for c,n in sorted(amounts_over_time.items(), key=lambda kv: kv[1][-1], reverse=True)[:10]]
    other_chems = [c for c in amounts_over_time.keys() if c not in top_chems]

    for chem in other_chems:
        amounts = amounts_over_time[chem]
        plt.plot(amounts, c="0.5", alpha=0.5)
    for chem in top_chems:
        amounts = amounts_over_time[chem]
        label = chem.designation
        color = None  # let it assign them a color
        plt.plot(amounts, label=label, c=color)

    plt.legend(loc="upper left")
    plt.show()

    plt.plot(temperature_over_time)
    plt.show()


def react_one_pair_from_many_chemicals(chems, amounts, temperature):
    # pick a pair based on the amounts (common chemicals are more likely to bump into each other)
    p = np.array([amounts[c] for c in chems]) / sum(amounts.values())
    ns = list(range(len(chems)))
    ci1 = np.random.choice(ns, p=p)
    ci2 = np.random.choice(ns, p=p)
    c1 = chems[ci1]
    c2 = chems[ci2]
    n1 = amounts[c1]
    n2 = amounts[c2]
    reaction = c1.get_reaction_properties(c2)

    if reaction["possible"]:
        # higher temperature decreases repulsion
        repulsion = reaction["repulsion"]
        effective_repulsion = reaction["repulsion"] - temperature
        # print(f"repulsion {repulsion}, temperature {temperature:.4f}, effective repulsion {effective_repulsion:.4f}")

        # sigmoid is probability of reaction occurring
        # lower repulsion = higher probability, so use 1/(1+exp(+x)) not -x
        try:
            z = 1 / (1 + math.exp(effective_repulsion))
        except OverflowError:
            if effective_repulsion > 0:
                z = 0
            elif effective_repulsion < 0:
                z = 1
            else:
                raise ValueError(z)

        r = random.random()
        reaction_happens = r < z
        # print(f"z {z:.4f}, r {r:.4f}, reaction happens {reaction_happens}")
    else:
        reaction_happens = False

    if reaction_happens:
        n1_needed = reaction["amount of first reactant"]
        n2_needed = reaction["amount of second reactant"]
        reaction_happens = n1 >= n1_needed and n2 >= n2_needed

    if reaction_happens:
        # print(f"picked {c1.designation} ({n1} units) and {c2.designation} ({n2} units)")
        # print(reaction)

        initial_n_particles = sum(amounts.values())
        initial_energy = temperature * initial_n_particles  # temperature is average kinetic energy

        # do one of the reaction
        n1_used = n1_needed
        n2_used = n2_needed
        c3 = c1.react_with(c2)
        n3_made = reaction["amount of product"]

        amounts[c1] -= n1_used
        amounts[c2] -= n2_used
        if c3 not in amounts:
            amounts[c3] = 0
        amounts[c3] += n3_made

        # remove used-up chemicals
        chems = [c for c in chems if amounts[c] != 0]
        amounts = {c: amounts[c] for c in chems}

        # print(f"-{n1_used} {c1.designation} , -{n2_used} {c2.designation} , +{n3_made} {c3.designation}")

        # energy diff is already scaled by amounts of reactants and product
        final_energy = initial_energy - reaction["energy_diff"]
        final_n_particles = sum(amounts.values())
        temperature = final_energy / final_n_particles

    else:
        pass

    return chems, amounts, temperature



if __name__ == "__main__":
    n_chems = 50
    all_path_strs = OrganicChemical.get_all_path_strs_in_order()
    path_strs = [next(all_path_strs) for i in range(n_chems)]
    chems = [OrganicChemical.from_path_str(s) for s in path_strs]
    # plot_reaction_types(chems)
    amounts = {chem: random.randint(0, 100) for chem in chems}
    temperature = 0
    react_many_chemicals(chems, amounts, temperature)


# how should chemicals behave? some numbers can determine whether / how often they will react with each other
# reactions should produce a result that has some new properties based on the reactants
# environment that organisms live in should have some effect on both the organism and the chemicals
# e.g. heat can speed up the decomposition of chemicals, aid metabolism, trigger certain reactions, etc.

# horizontal and vertical charges should be preserved, so that creates stoichiometry
# energy need not be preserved, treat this as endo/exothermic
# size need not be preserved, don't worry about this for now



