# try idea for generating land and water
# for each point on a grid, decide whether it should be land or water
# want result to look realistic, what to do?
# if no neighbors, 50/50
# if neighbors, weight probability so land is more likely to be next to land and water is more likely to be next to water
# start by doing this on large scale, like every 16th pixel in both dimensions (but treat the pixel 16 away as a neighbor since you want large-scale structure like continents and oceans to be possible)
# then zoom in and use same logic for every 8th, 4th, etc.
# or better yet, do a n*n grid with all points, then expand it and fill in the gaps


import random
import numpy as np
import matplotlib.pyplot as plt

WATER = 0
UNDECIDED = 0.5
LAND = 1

def decide(p_land):
    return LAND if random.random() < p_land else WATER

def get_probability_of_land(neighbors):
    # return np.mean(neighbors)  # allows all-land or all-sea to be too likely
    # try exponential decay of probability the more water there is, and make it symmetric
    x = np.mean(neighbors)
    return prob_func(x)

def prob_func(x):
    if x > 0.5:
        switched = True
        x = 1 - x
    else:
        switched = False
    # f(0.5) = 0.5
    # exponential decay with p getting cut in half (=0.25) with one land neighbor changed to water, so f(3/8) = 0.25
    assert 0 <= x <= 0.5
    half_lives_down = (0.5 - x) / (1/2 - 3/8)
    decay_factor = 2 ** (-1 * half_lives_down)
    p0 = 0.5
    p = p0 * decay_factor
    if switched:
        return 1 - p
    else:
        return p

def plot_prob_func():
    xs = np.arange(0, 1, 0.01)
    ys = []
    for x in xs:
        ys.append(prob_func(x))
    plt.plot(xs, ys)
    plt.show()

def decide_point(neighbors):
    # weight all neighbors equally, including UNDECIDED ones
    p_land = get_probability_of_land(neighbors)
    assert 0 <= p_land <= 1
    return decide(p_land)

class Grid:
    def __init__(self, side_length):
        self.side_length = side_length
        self.reset_grid()
        self.reset_undecided_points()

    def reset_grid(self):
        self.grid = np.full(shape=(self.side_length, self.side_length), fill_value = UNDECIDED)

    def reset_undecided_points(self):
        self.undecided_points = set((i, j) for i in range(self.side_length) for j in range(self.side_length))

    def fill(self):
        while len(self.undecided_points) > 0:
            coords = random.choice(list(self.undecided_points))
            neighbors = self.get_neighbor_values(coords)
            val = decide_point(neighbors)
            # print("{} => {}".format(coords, val))
            x, y = coords
            self.grid[x, y] = val
            self.undecided_points.remove(coords)

    def get_neighbor_values(self, coords):
        neighbor_coords = self.get_neighbor_coords(coords)
        return [self.grid[x, y] for x, y in neighbor_coords]

    def get_neighbor_coords(self, coords):
        # 8, not 4
        is_valid = lambda x, y: 0 <= x < self.side_length and 0 <= y < self.side_length
        x, y = coords
        offsets = [-1, 0, 1]
        cands = [(x + i, y + j) for i in offsets for j in offsets if i != 0 or j != 0]
        return [c for c in cands if is_valid(*c)]

    def expand(self, factor=2):
        assert int(factor) == factor, "expansion factor must be int"
        old_side_length = self.side_length
        old_grid = self.grid
        self.side_length = self.side_length * factor
        self.reset_grid()
        self.reset_undecided_points()
        for x in range(old_side_length):
            for y in range(old_side_length):
                # one way was to just scale up the single point and leave the spaces in between undecided, but ended up too random-looking
                # try scaling up the point in area too, then "redoing" every point, so the boundaries will blur
                new_x0 = x * factor
                new_y0 = y * factor
                offsets = [i for i in range(factor)]
                for offset_x in offsets:
                    for offset_y in offsets:
                        new_x = new_x0 + offset_x
                        new_y = new_y0 + offset_y
                        self.grid[new_x, new_y] = old_grid[x, y]
                        # self.undecided_points.remove((new_x, new_y))
                        # re-decide all points

    def plot(self):
        plt.imshow(self.grid)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    # plot_prob_func()
    g = Grid(4)
    g.fill()
    for i in range(5):
        print("step {}".format(i))
        g.expand()
        g.fill()
    g.plot()
