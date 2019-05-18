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

def decide_point(neighbors):
    # weight all neighbors equally, including UNDECIDED ones
    p_land = np.mean(neighbors)
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
        for i in range(old_side_length):
            for j in range(old_side_length):
                new_i = i * factor
                new_j = j * factor
                self.grid[new_i, new_j] = old_grid[i, j]
                self.undecided_points.remove((i, j))

    def plot(self):
        plt.imshow(self.grid)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    g = Grid(2)
    g.fill()
    for _ in range(5):
        g.expand()
        g.fill()
        g.plot()
