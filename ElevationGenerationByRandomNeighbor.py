import random
import numpy as np
import matplotlib.pyplot as plt


def make_none_array(shape):
    a = np.empty(shape, dtype=float)
    a[:] = np.nan
    return a


class Map:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.array = make_none_array((x_size, y_size))
        self.filled_positions = set()
        self.unfilled_neighbors = set()
        self.unfilled_inaccessible = set()
        self.initialize_unfilled_inaccessible()

    def initialize_unfilled_inaccessible(self):
        for x in range(self.x_size):
            for y in range(self.y_size):
                self.unfilled_inaccessible.add((x, y))

    def fill_position(self, x, y, value):
        assert (x, y) not in self.filled_positions
        assert self.filled_positions & self.unfilled_neighbors == set()
        assert self.filled_positions & self.unfilled_inaccessible == set()
        assert self.unfilled_neighbors & self.unfilled_inaccessible == set()
        self.array[x, y] = value
        self.filled_positions.add((x, y))
        self.unfilled_neighbors -= {(x, y)}
        self.unfilled_inaccessible -= {(x, y)}
        for neighbor_coords in self.get_neighbors(x, y):
            if neighbor_coords in self.unfilled_inaccessible:
                assert neighbor_coords not in self.unfilled_neighbors
                self.unfilled_neighbors.add(neighbor_coords)
                self.unfilled_inaccessible -= {neighbor_coords}

    def get_value_at_position(self, x, y):
        return self.array[x, y]

    def get_neighbors(self, x, y):
        res = set()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x = x + dx
                new_y = y + dy
                if new_x < 0 or new_x >= self.x_size:
                    continue
                if new_y < 0 or new_y >= self.y_size:
                    continue
                res.add((new_x, new_y))
        return res

    def add_random_zero_loop(self):
        x = random.randrange(self.x_size)
        y = random.randrange(self.y_size)
        trajectory_0 = [(x, y)]
        trajectory_1 = [(x, y)]
        # display_arr = make_none_array((self.x_size, self.y_size))  # for debugging
        # display_arr = display_arr.astype(str)
        # display_arr[x, y] = "S"
        while True:
            # two ants walking away from starting point, can pass through themselves
            # when they meet, cut off the paths and take the trajectories up to that point
            neighbors_0 = self.get_neighbors(*trajectory_0[-1])
            neighbors_1 = self.get_neighbors(*trajectory_1[-1])
            coords_0 = random.choice(list(neighbors_0))
            coords_1 = random.choice(list(neighbors_1))
            # print(coords_0, coords_1)
            if coords_0 in trajectory_1:
                cut_point = coords_0
                trajectory_0.append(coords_0)
                # display_arr[coords_0] = "0"
                break
            elif coords_1 in trajectory_0:
                cut_point = coords_1
                trajectory_1.append(coords_1)
                # display_arr[coords_1] = "1"
                break
            trajectory_0.append(coords_0)
            # display_arr[coords_0] = "0"
            trajectory_1.append(coords_1)
            # display_arr[coords_1] = "1"
            # print(display_arr)
            # input("debug")

        
        cut_index_0 = trajectory_0.index(cut_point)
        cut_index_1 = trajectory_1.index(cut_point)
        points = set(trajectory_0[:cut_index_0]) | set(trajectory_1[:cut_index_1]) | {cut_point}
        for p in points:
            self.fill_position(p[0], p[1], 0)

    def fill_elevations(self):
        while len(self.unfilled_neighbors) + len(self.unfilled_inaccessible) > 0:
            p = random.choice(list(self.unfilled_neighbors))
            # neighbor = random.choice([x for x in self.get_neighbors(*p) if x in self.filled_positions])
            # neighbor_el = self.get_value_at_position(*neighbor)
            filled_neighbors = [x for x in self.get_neighbors(*p) if x in self.filled_positions]
            average_neighbor_el = np.mean([self.get_value_at_position(*n) for n in filled_neighbors])
            d_el = np.random.normal(0, 100)
            el = average_neighbor_el + d_el
            self.fill_position(p[0], p[1], el)

    def plot(self):
        print(self.array)
        plt.imshow(self.array)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    m = Map(250, 250)
    m.add_random_zero_loop()
    m.fill_elevations()
    m.plot()
