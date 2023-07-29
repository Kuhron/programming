# idea for an implementation of an ecosystem
# creatures can reproduce and place offspring near them in N dimensions (let N=3 for ease for now)
# later we can visualize where the organisms are by using MDS, and/or 3D plot
# organisms have ways they can interact when they encounter another organism (fight, eat, run, have sex, ignore, etc.)
# try making it easy to calculate stuff here, e.g. an easy way to check if there are organisms nearby (not querying a tree or something)

# use L1 distance and discrete grid of locations at integer coordinates
# so organism can see/detect within a certain distance


import numpy as np


class SparseGrid:
    def __init__(self):
        self.d = {}

    def move_item(item, old_position, new_position):
        self.remove_item(item, old_position)
        self.add_item(item, new_position)

    def add_item(self, item, position):
        if position not in self.d:
            self.d[position] = set()
        self.d[position].add(item)

    def remove_item(self, item, position):
        self.d[position].remove(item)
        if len(self.d[position]) == 0:
            del self.d[position]

    def __getitem__(self, position):
        return self.d.get(position, set())

    def __setitem__(self, position, item):
        self.add_item(item, position)


class Organism:
    def __init__(self):
        self.location = None

    def place(self, location):
        self.location = location

    def reproduce(self, grid):
        new_location = tuple(x + random.randint(-1, 1) for x in self.location)

def get_points_at_distance(p, d):
    assert type(p) is tuple
    assert len(p) > 0, "empty tuple"
    assert all(type(x) is int for x in p), set(type(x) for x in p)
    assert type(d) is int, type(d)
    assert d >= 0, d

    if d == 0:
        return [p]
    a, *ps = p
    if len(p) == 1:
        return [(a-d,), (a+d,)]
    l = []
    for x in range(-d, d+1):
        l += [(a+x,) + p2 for p2 in get_points_at_distance(p[1:], d - abs(x))]

    # debug
    # for p2 in l:
    #     s = 0
    #     for i in range(len(p)):
    #         s += abs(p[i] - p2[i])
    #     assert s == d, f"point with wrong distance: {p} to {p2} is {s} but should be {d}"

    return l


if __name__ == "__main__":
    grid = SparseGrid()

    org = Organism()
    org.run_routine(grid)

