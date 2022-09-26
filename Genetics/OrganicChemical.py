import numpy as np
import matplotlib.pyplot as plt
import math


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
        return f"{self.name} : {self.path_str}" if self.has_name() else self.path_str

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
            behavior = ("easy" if repulsion <= 0 else "hard") + "-" + ("hot" if energy_diff <= 0 else "cold")
        else:
            behavior = "none"

        if possible:
            assert horizontality_diff == 0, "stoichiometry error"
            assert verticality_diff == 0, "stoichiometry error"

        return {
            "possible": possible,
            "behavior": behavior,
            "repulsion": repulsion,
            "energy_diff": energy_diff,
            "size_diff": size_diff,
            "amount of first reactant": xt,
            "amount of second reactant": yt,
            "amount of product": zt,
        }



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


# how should chemicals behave? some numbers can determine whether / how often they will react with each other
# reactions should produce a result that has some new properties based on the reactants
# environment that organisms live in should have some effect on both the organism and the chemicals
# e.g. heat can speed up the decomposition of chemicals, aid metabolism, trigger certain reactions, etc.

# horizontal and vertical charges should be preserved, so that creates stoichiometry
# energy need not be preserved, treat this as endo/exothermic
# size need not be preserved, don't worry about this for now
