import numpy as np
import matplotlib.pyplot as plt


class OrganicChemical:
    properties = {
        "leftness": "N", "rightness": "N", "horizontality": "Z",
        "downness": "N", "upness": "N", "verticality": "Z",
        "mass": "R", "size": "R", "density": "R",
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
            mass, size, density):
        self.path = path
        self.path_tuple = to_tuple(path)
        self.path_str = self.get_path_str()
        self.leftness = leftness
        self.rightness = rightness
        self.horizontality = horizontality
        self.downness = downness
        self.upness = upness
        self.verticality = verticality
        self.mass = mass
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
        mass = np.mean(distances)
        size = max(distances)
        density = mass / size if size != 0 else 0
        leftness = min(xs) * -1
        rightness = max(xs)
        horizontality = rightness - leftness
        downness = min(ys) * -1
        upness = max(ys)
        verticality = upness - downness

        return OrganicChemical(path=path,
            leftness=leftness, rightness=rightness, horizontality=horizontality,
            downness=downness, upness=upness, verticality=verticality,
            mass=mass, size=size, density=density,
        )

    def get_path_str(self):
        s = ""
        x0, y0 = 0, 0
        for x1, y1 in self.path[1:]:
            dx = x1-x0
            dy = y1-y0
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
        if self.has_name():
            s += self.name + ": "
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
        s += "]"
        return s

    def __repr__(self):
        return self.to_str()




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

