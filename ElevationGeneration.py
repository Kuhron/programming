import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
from PIL import Image
from scipy import ndimage
from datetime import datetime, timedelta


def make_nan_array(shape):
    a = np.empty(shape, dtype=float)
    a[:] = np.nan
    return a

def make_blank_condition_array(shape):
    a = np.empty(shape, dtype=object)
    a[:] = lambda x: True
    return a


def elevation_change_parabolic(d, max_d, max_change):
    b = max_d
    h = max_change
    return -h/(b**2) * (d**2) + 2*h/b * d

def elevation_change_linear(d, max_d, max_change):
    b = max_d
    h = max_change
    return h/b * d

def elevation_change_semicircle(d, max_d, max_change):
    b = max_d
    h = max_change
    return h/b * np.sqrt(2*b*d - d**2)

def elevation_change_inverted_semicircle(d, max_d, max_change):
    b = max_d
    h = max_change
    return h - h/b * np.sqrt(b**2 - d**2)

def elevation_change_sinusoidal(d, max_d, max_change):
    b = max_d
    h = max_change
    return h/2 * (1 + np.sin(np.pi/b * (d - b/2)))

def elevation_change_constant(d, max_d, max_change):
    return max_change

ELEVATION_CHANGE_FUNCTIONS = [
    elevation_change_linear,  # default to linear, should be fastest non-constant
    # elevation_change_parabolic,
    # elevation_change_semicircle,  # changes too fast near 0, makes very steep slopes too often
    # elevation_change_inverted_semicircle,  # slow
    # elevation_change_sinusoidal,  # slow
    # elevation_change_constant,  # f(0) is not 0, makes cliffs
]

def show_elevation_change_functions():
    xs = np.linspace(0, 1, 20)
    max_x = 1
    max_change = 1
    for i, f in enumerate(ELEVATION_CHANGE_FUNCTIONS):
        ys = [f(x, max_x, max_change) for x in xs]
        plt.plot(xs, ys)
        plt.title("function index {}".format(i))
        plt.show()


def get_land_and_sea_colormap():
    # see PrettyPlot.py
    linspace_cmap_forward = np.linspace(0, 1, 128)
    linspace_cmap_backward = np.linspace(1, 0, 128)
    blue_to_black = mcolors.LinearSegmentedColormap.from_list('BlBk', [
        mcolors.CSS4_COLORS["blue"], 
        mcolors.CSS4_COLORS["black"],
    ])
    land_colormap = mcolors.LinearSegmentedColormap.from_list('land', [
        mcolors.CSS4_COLORS["darkgreen"],
        mcolors.CSS4_COLORS["limegreen"],
        mcolors.CSS4_COLORS["gold"],
        mcolors.CSS4_COLORS["darkorange"],
        mcolors.CSS4_COLORS["red"],
        mcolors.CSS4_COLORS["saddlebrown"],
        mcolors.CSS4_COLORS["gray"],
        mcolors.CSS4_COLORS["white"],
        # mcolors.CSS4_COLORS[""],
    ])
    # colors_land = plt.cm.YlOrBr(linspace_cmap_backward)  # example of how to call existing colormap object
    colors_land = land_colormap(linspace_cmap_forward)
    colors_sea = blue_to_black(linspace_cmap_backward)
    colors = np.vstack((colors_sea, colors_land))
    colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return colormap


class Map:
    def __init__(self, x_size, y_size, array=None):
        # really row size and column size, respectively
        self.x_size = x_size
        self.y_size = y_size
        self.x_range = np.arange(self.x_size)
        self.y_range = np.arange(self.y_size)
        if array is None:
            self.array = make_nan_array((x_size, y_size))
        else:
            self.array = array
        self.condition_array = make_blank_condition_array((x_size, y_size))
        #self.filled_positions = set()
        #self.unfilled_neighbors = set()
        #self.unfilled_inaccessible = set()
        #self.initialize_unfilled_inaccessible()
        self.frozen_points = set()
        self.neighbors_memoized = {}
        self.memoize_all_neighbors()
        # self.untouched_points = self.get_all_points()

    #def initialize_unfilled_inaccessible(self):
    #    for x in range(self.x_size):
    #        for y in range(self.y_size):
    #            self.unfilled_inaccessible.add((x, y))

    def get_all_points(self):
        return {(x, y) for x in range(self.x_size) for y in range(self.y_size)}

    # def touch(self, x, y):
    #     self.untouched_points -= {(x, y)}

    # def untouch_all_unfrozen_points(self):
    #     self.untouched_points = self.get_all_points() - self.frozen_points

    def is_corner_pixel(self, x, y):
        return x in [0, self.x_size-1] and y in [0, self.y_size-1]

    def is_edge_pixel(self, x, y):
        x_edge = x in [0, self.x_size-1]
        y_edge = y in [0, self.y_size-1]
        return (x_edge and not y_edge) or (y_edge and not x_edge)

    def is_interior_pixel(self, x, y):
        return 1 <= x < self.x_size-1 and 1 <= y < self.y_size-1

    def get_representative_pixel(self, x, y):
        # for corner pixel, return itself since all 4 have different neighbor shape
        # for edge pixel, return one of the pixels on that edge
        # for interior pixel, return (1, 1)
        # then memoize only a total of 9 neighbor arrays for any size image, and just add offset
        x_edge = x in [0, self.x_size-1]
        y_edge = y in [0, self.y_size-1]
        if x_edge and y_edge:
            # corner
            return (x, y)
        elif x_edge or y_edge:
            # but not both, as that would have been caught by the previous condition
            # edge, replace the interior coordinate with 1
            if x_edge:
                return (x, 1)
            if y_edge:
                return (1, y)
        else:
            # interior
            return (1, 1)

    def memoize_all_neighbors(self):
        # just calling get_neighbors will memoize
        # corner pixels
        for x, y in (0, 0), (0, self.y_size-1), (self.x_size-1, 0), (self.x_size, self.y_size):
            n = self.get_neighbors(x, y)
        # edge representatives
        for x, y in (0, 1), (1, 0), (1, self.y_size-1), (self.x_size-1, 1):
            n = self.get_neighbors(x, y)
        # interior representative
        n = self.get_neighbors(1, 1)

        # old way, memory hog
        # for x in range(self.x_size):
        #     for y in range(self.y_size):
        #         n = self.get_neighbors(x, y)
        #         # function will memoize them
        # assert len(self.neighbors_memoized) == self.size()

    def size(self):
        return self.x_size * self.y_size

    def is_valid_point(self, x, y):  # flagged as slow due to sheer number of calls
        return 0 <= x < self.x_size and 0 <= y < self.y_size

    def filter_invalid_points(self, iterable):
        res = set()
        for p in iterable:
            if self.is_valid_point(*p):
                res.add(p)
        return res

    def freeze_point(self, x, y):
        self.frozen_points.add((x, y))
        # self.touch(x, y)

    def unfreeze_point(self, x, y):
        self.frozen_points.remove((x, y))

    def add_condition_at_position(self, x, y, func):
        assert callable(func)
        self.condition_array[x, y] = func

    def new_value_satisfies_condition(self, x, y, value):
        condition = self.condition_array[x, y]
        if callable(condition):
            res = condition(value)
            assert type(res) in [bool, np.bool_], "invalid condition return value at {} for value {}: {} of type {}".format((x, y), value, res, type(res))
            return res
        else:
            raise ValueError("invalid condition type {}".format(type(condition)))

    def fill_position(self, x, y, value):
        assert (x, y) not in self.frozen_points, "can't change frozen point {}".format((x, y))
        self.array[x, y] = value
        # self.touch(x, y)

        # old way
        # raise Exception("do not use")
        # assert (x, y) not in self.filled_positions
        # assert self.filled_positions & self.unfilled_neighbors == set()
        # assert self.filled_positions & self.unfilled_inaccessible == set()
        # assert self.unfilled_neighbors & self.unfilled_inaccessible == set()
        # self.array[x, y] = value
        # self.filled_positions.add((x, y))
        # self.unfilled_neighbors -= {(x, y)}
        # self.unfilled_inaccessible -= {(x, y)}
        # for neighbor_coords in self.get_neighbors(x, y):
        #     if neighbor_coords in self.unfilled_inaccessible:
        #         assert neighbor_coords not in self.unfilled_neighbors
        #         self.unfilled_neighbors.add(neighbor_coords)
        #         self.unfilled_inaccessible -= {neighbor_coords}

    def fill_point_set(self, point_set, value):
        for p in point_set:
            if p not in self.frozen_points:
                self.fill_position(p[0], p[1], value)

    def fill_all(self, value):
        for x in range(self.x_size):
            for y in range(self.y_size):
                self.fill_position(x, y, value)

    def get_value_at_position(self, x, y):
        return self.array[x, y]

    def get_neighbors(self, x, y, mode=8):
        if mode == 4:
            res = set()
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                p = (x+dx, y+dy)
                if self.is_valid_point(*p):
                # if 0 <= x+dx < self.x_size and 0 <= y+dy < self.y_size:
                    res.add(p)
            return res

        elif mode == 8:
            rep = self.get_representative_pixel(x, y)
            offset = (x - rep[0], y - rep[1])
            if rep in self.neighbors_memoized:
                rep_neighbors = self.neighbors_memoized[rep]
                return {(n[0] + offset[0], n[1] + offset[1]) for n in rep_neighbors}  # flagged as slow: setcomp
    
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
            self.neighbors_memoized[(x, y)] = res
            return res
        else:
            raise ValueError("unknown mode {}".format(mode))

    def get_random_path(self, a, b, points_to_avoid):
        # start and end should inch toward each other
        i = 0
        points_in_path = {a, b}
        current_a = a
        current_b = b
        while True:
            which_one = i % 2
            current_point = [current_a, current_b][which_one]
            objective = [current_b, current_a][which_one]
            points_to_avoid_this_step = points_to_avoid | points_in_path
            
            next_step = self.get_next_step_in_path(current_point, objective, points_to_avoid_this_step)
            if which_one == 0:
                current_a = next_step
            elif which_one == 1:
                current_b = next_step
            else:
                raise

            points_in_path.add(next_step)

            if current_a in self.get_neighbors(*current_b):
                break

            i += 1
        return points_in_path

    def get_next_step_in_path(self, current_point, objective, points_to_avoid):
        dx = objective[0] - current_point[0]
        dy = objective[1] - current_point[1]
        z = dx + dy*1j
        objective_angle = np.angle(z, deg=True)
        neighbors = self.get_neighbors(*current_point)
        neighbors = [n for n in neighbors if n not in points_to_avoid]
        neighbor_angles = [np.angle((n[0]-current_point[0]) + (n[1]-current_point[1])*1j, deg=True) for n in neighbors]
        # set objective angle to zero for purposes of determining weight
        neighbor_effective_angles = [abs(a - objective_angle) % 360 for a in neighbor_angles]
        neighbor_effective_angles = [a-360 if a>180 else a for a in neighbor_effective_angles]
        neighbor_weights = [180-abs(a) for a in neighbor_effective_angles]
        assert all(0 <= w <= 180 for w in neighbor_weights), "{}\n{}\n{}".format(neighbors, neighbor_effective_angles, neighbor_weights)
        total_weight = sum(neighbor_weights)
        norm_weights = [w / total_weight for w in neighbor_weights]
        chosen_one_index = np.random.choice([x for x in range(len(neighbors))], p=norm_weights)
        chosen_one = neighbors[chosen_one_index]
        return chosen_one

    def get_random_contiguous_region(self, x=None, y=None, expected_size=None, points_to_avoid=None, prioritize_internal_unfilled=False):
        assert expected_size > 1, "invalid expected size {}".format(expected_size)
        if points_to_avoid is None:
            points_to_avoid = set()
        points_to_avoid |= self.frozen_points
        center = (x, y)
        while center is (None, None) or center in points_to_avoid:
            center = self.get_random_point()
        neighbors = [p for p in self.get_neighbors(*center) if p not in points_to_avoid]
        size = -1
        while size < 1:
            size = int(np.random.normal(expected_size, expected_size/2))
        # print("making region of size {}".format(size))

        radius = int(round(np.sqrt(size/np.pi)))
        circle = self.get_circle_around_point(center[0], center[1], radius, barrier_points=points_to_avoid)
        return circle

        # this way looks better but is slow
        # raise   
        # points = {center}
        # for step in range(size-1):
        #     if len(neighbors) == 0:
        #         break
        #     if prioritize_internal_unfilled:
        #         n_filled_neighbors_of_neighbors = [sum(n2 in points for n2 in self.get_neighbors(*n)) for n in neighbors]
        #         weights = [x/sum(n_filled_neighbors_of_neighbors) for x in n_filled_neighbors_of_neighbors]
        #         chosen_index = np.random.choice(list(range(len(neighbors))), p=weights)
        #         current_point = neighbors[chosen_index]
        #     else:
        #         current_point = random.choice(neighbors)
        #     assert current_point not in points, "duplicating point in contiguous region selection"
        #     points.add(current_point)
        #     neighbors.remove(current_point)
        #     new_neighbors = self.get_neighbors(*current_point) - points - points_to_avoid
        #     neighbors = list(set(neighbors) | set(new_neighbors))
        # # assert len(points) == size, "planned output size {}, got {}".format(size, len(points))
        # return points

    def get_circle_around_point(self, x, y, radius, barrier_points=None):  # flagged as slow, look for existing algorithms
        # print("\ncenter {}\nbarrier points\n{}".format((x, y), sorted(barrier_points)))
        # input("please debug")
        if barrier_points is None:
            barrier_points = set()
        assert (x, y) not in barrier_points, "can't make circle with center in barrier"
        points = {(x, y)}
        def get_max_dy(dx):
            return int(round(np.sqrt(radius**2-dx**2)))

        # can tell which points are on inside vs outside of barrier wall by doing this:
        # starting in center, go left/right and count barrier crossings (+= 0.5 when is_in_barrier_set changes truth value)
        # then go up/down from there and continue counting
        # so each point in the set is associated with a number of barrier crossings
        # those ending in 0.5 are in the barrier itself, those == 1 mod 2 are on the other side
        # so take those == 0 mod 2

        barrier_crossings_by_point = {(x, y): 0}
        
        # print("\nstarting circle around {}".format((x, y)))
        # start with dx = 0 to create the central line
        dy = 0
        max_dx = radius
        for direction in [-1, 1]:
            # reset barrier crossings to center's value
            n_barrier_crossings = barrier_crossings_by_point[(x, y)]
            last_was_on_barrier = False
            for abs_dx in range(max_dx+1):
                dx = direction * abs_dx
                this_x = x + dx
                this_y = y + dy
                # print("adding central line point at {}".format((this_x, this_y)))
                if not self.is_valid_point(this_x, this_y):
                    continue
                is_on_barrier = (this_x, this_y) in barrier_points
                if is_on_barrier != last_was_on_barrier:
                    n_barrier_crossings += 0.5
                barrier_crossings_by_point[(this_x, this_y)] = n_barrier_crossings
                last_was_on_barrier = is_on_barrier

        # now do the rest
        for dx in range(-radius, radius+1):
            this_x = x + dx
            central_axis_point = (x, y)
            assert central_axis_point in barrier_crossings_by_point, "can't find central axis point {}".format(central_axis_point)
            max_dy = get_max_dy(dx)
            for direction in [-1, 1]:
                n_barrier_crossings = barrier_crossings_by_point[central_axis_point]
                last_was_on_barrier = central_axis_point in barrier_points
                for abs_dy in range(max_dy+1):
                    dy = direction * abs_dy
                    this_y = y + dy
                    # print("adding non-central point at {}".format((this_x, this_y)))
                    if not self.is_valid_point(this_x, this_y):
                        continue
                    is_on_barrier = (this_x, this_y) in barrier_points
                    if is_on_barrier != last_was_on_barrier:
                        n_barrier_crossings += 0.5
                    barrier_crossings_by_point[(this_x, this_y)] = n_barrier_crossings
                    last_was_on_barrier = is_on_barrier

        res = {p for p, n in barrier_crossings_by_point.items() if n % 2 == 0}
        # print("returning:\n{}".format(res))

        # for dx in range(-radius, radius+1):
        #     max_dy = get_max_dy(dx)
        #     for ?
        # starting_set = {(x+dx, y+dy) for dx in range(-radius, radius+1) for dy in get_dy_range(dx)}
        # print("before barrier len = {}".format(len(res)))
        # res = self.apply_barrier(res, barrier_points, x, y)
        # print(" after barrier len = {}".format(len(res)))
        return res

    # def apply_barrier(self, point_set, barrier_points, center_x, center_y):
    #     # very slow!!
    #     # start from center and emanate outward until can't anymore
    #     # don't want it to jump past diagonal borders, so do horizontal/vertical neighbors only for this emanation
    #     points = {(center_x, center_y)}
    #     previous_points = {(center_x, center_y)}
    #     while True:
    #         # print("prev", sorted(previous_points))
    #         # input("a")
    #         horizvert_neighbors = set()
    #         for p in previous_points:
    #             ns = {n for n in self.get_neighbors(p[0], p[1], mode=4) if n in point_set and n not in barrier_points | points}
    #             horizvert_neighbors |= ns
    #         # print("hvn", sorted(horizvert_neighbors))
    #         # input("a")
    #         if len(horizvert_neighbors) == 0:
    #             break
    #         points |= horizvert_neighbors
    #         # print("{} pts".format(len(points)))
    #         # input("a")
    #         previous_points = horizvert_neighbors
    #     return points

    def get_distances_from_edge(self, point_set, use_scipy_method=True):
        if use_scipy_method:
            min_x = np.inf
            max_x = -np.inf
            min_y = np.inf
            max_y = -np.inf
            for p in point_set:
                x, y = p
                min_x = min(x, min_x)
                max_x = max(x, max_x)
                min_y = min(y, min_y)
                max_y = max(y, max_y)
            
            # put them in an array
            to_relative_coords = lambda p: (p[0]-min_x, p[1]-min_y)
            to_absolute_coords = lambda p: (p[0]+min_x, p[1]+min_y)
            arr_x_size = max_x - min_x + 1
            arr_y_size = max_y - min_y + 1
            arr = np.zeros((arr_x_size, arr_y_size))
            rels = {}
            for p in point_set:
                xrel, yrel = to_relative_coords(p)
                rels[p] = (xrel, yrel)
                arr[xrel, yrel] = 1
            distance_transform_matrix = ndimage.morphology.distance_transform_edt(arr)
            res = {}
            for p in point_set:
                xrel, yrel = rels[p]
                d = distance_transform_matrix[xrel, yrel]
                res[p] = d - 1
            return res

        else:
            # old way, slower than scipy
            if len(point_set) == 0:
                return {}
            res = {}
            points_on_edge = [p for p in point_set if any(n not in point_set for n in self.get_neighbors(*p))]  # flagged as slow: genexpr
            assert len(points_on_edge) > 0, "point set has no edge members:\n{}".format(sorted(point_set))
            for p in points_on_edge:
                res[p] = 0
            interior_point_set = point_set - set(points_on_edge)
            if len(interior_point_set) > 0:
                interior_distances = self.get_distances_from_edge(interior_point_set, use_scipy_method=False)
                for p, d in interior_distances.items():
                    res[p] = d + 1
            return res

    def make_random_elevation_change(self, expected_size, positive_feedback=False):
        center = self.get_random_point()
        # center = self.get_random_untouched_point()
        # print("making change at {}".format(center))
        changing_reg = self.get_random_contiguous_region(center[0], center[1], expected_size=expected_size, points_to_avoid=self.frozen_points)
        changing_reg_size = len(changing_reg)
        # print("expected size {}, got {}".format(expected_size, changing_reg_size))
        changing_reg_center_of_mass = (
            int(round(1/changing_reg_size * sum(p[0] for p in changing_reg))),
            int(round(1/changing_reg_size * sum(p[1] for p in changing_reg)))
        )
        e_center_of_mass = self.array[changing_reg_center_of_mass[0], changing_reg_center_of_mass[1]]
        reference_x, reference_y = changing_reg_center_of_mass
        # radius_giving_equivalent_area = np.sqrt(changing_reg_size/np.pi)
        radius_giving_expected_area = np.sqrt(expected_size/np.pi)
        desired_area_ratio_at_sea_level = 5
        desired_area_ratio_at_big_abs = 0.1
        big_abs = 1000

        # try to get mountain chains to propagate:
        # if center point is low abs, look at bigger region, might catch mountain
        # if center point is high abs, look at smaller region, don't let lowland water it down
        if abs(e_center_of_mass) >= big_abs:
            desired_area_ratio = desired_area_ratio_at_big_abs
        else:
            slope = (desired_area_ratio_at_big_abs - desired_area_ratio_at_sea_level) / (big_abs - 0)
            desired_area_ratio = desired_area_ratio_at_sea_level + slope * abs(e_center_of_mass)
        # radius = int(round(radius_giving_equivalent_area * np.sqrt(desired_area_ratio)))
        radius = int(round(radius_giving_expected_area * np.sqrt(desired_area_ratio)))
        reference_reg = self.get_circle_around_point(reference_x, reference_y, radius=radius)
        # reference_reg = changing_reg
        distances = self.get_distances_from_edge(changing_reg)
        max_d = max(distances.values())
        if max_d == 0:
            raw_func = elevation_change_constant
        else:
            raw_func = random.choice(ELEVATION_CHANGE_FUNCTIONS)

        if positive_feedback:
            # land begets land, sea begets sea
            elevations_in_refreg = [self.array[p[0], p[1]] for p in reference_reg]
            e_avg = np.mean(elevations_in_refreg)
            e_max = np.max(elevations_in_refreg)  # for detecting mountain nearby, chain should propagate
            e_min = np.min(elevations_in_refreg)
            elevation_sign = (1 if e_avg > 0 else -1)
            big_signed = elevation_sign * big_abs
            critical_abs = 100  # above this abs, go farther in that direction until reach big_abs_elevation
            critical_signed = elevation_sign * critical_abs
            critical_excess = e_avg - critical_signed
            big_remainder = big_signed - e_avg
            mountain_or_trench_nearby = abs(e_max) >= big_abs or abs(e_min) >= big_abs
  
            mu = \
                0 if abs(e_avg) > big_abs else \
                10 if e_avg > critical_abs else \
                -10 if e_avg < -1*critical_abs else \
                0

            # if False: #mountain_or_trench_nearby:
            #     pass
            #     # try to propagate it in a line, i.e., the closer e_avg is to mountain size, the more likely it is to rise
            #     alpha_between_critical_and_big = (e_avg - critical_signed)/(big_signed - critical_signed)
            #     # closer to big = bigger alpha, bigger expected rise
            #     a = np.random.uniform(alpha_between_critical_and_big)
            #     mu = a * big_remainder
            # else:

            # try another idea, extreme elevations have expected movement of zero
            # but moderate ones move more in their direction
            # if abs(average_elevation_in_refreg) < abs(big_elevation_signed):
            #     mu = average_elevation_in_refreg
            # else:
            #     mu = 0

            # old correction to mu, I think this is causing overcorrection, might be responsible for mountain rings
            # because it sees mountains that are taller than big_abs, wants to drop them, ends up creating lowlands in the rest of the
            # changing region, could this cause a mountain ring to propagate toward shore, leaving central valley?
            # mu = average_elevation_in_refreg
            # if abs(mu) > big_abs_elevation:
            #     # decrease mu linearly down to 0 at 2*big_abs_elevation, and then drop more after that to decrease
            #     big_elevation_signed = big_abs_elevation * (1 if mu > 0 else -1)
            #     mu_excess = mu - big_elevation_signed
            #     mu -= 2*mu_excess

        else:
            mu = 0

        sigma = max(10, abs(e_avg)) if abs(e_avg) < big_abs else 10
        max_change = np.random.normal(mu, sigma)

        func = lambda d: raw_func(d, max_d, max_change)
        changes = {p: func(d) for p, d in distances.items() if p not in self.frozen_points}
        for p, d_el in changes.items():
            current_el = self.get_value_at_position(*p)
            new_el = current_el + d_el
            if self.new_value_satisfies_condition(p[0], p[1], new_el):
                self.fill_position(p[0], p[1], new_el)

    def get_random_point(self, border_width=0):
        x = random.randrange(border_width, self.x_size - border_width)
        y = random.randrange(border_width, self.y_size - border_width)
        return (x, y)

    # def get_random_untouched_point(self):
    #     return random.choice(list(self.untouched_points))

    def get_random_zero_loop(self):
        x0_0, y0_0 = self.get_random_point(border_width=2)
        dx = 0
        dy = 0
        while np.sqrt(dx**2 + dy**2) < self.x_size * 1/2 * np.sqrt(2):
            x0_1 = random.randrange(2, self.x_size - 2)
            y0_1 = random.randrange(2, self.y_size - 2)
            dx = x0_1 - x0_0
            dy = y0_1 - y0_0
        # x0_1 = (x0_0 + self.x_size // 2) % self.x_size  # force it to be considerably far away to get more interesting result
        # y0_1 = (y0_0 + self.y_size // 2) % self.y_size
        source_0 = (x0_0, y0_0)
        source_1 = (x0_1, y0_1)
        path_0 = self.get_random_path(source_0, source_1, points_to_avoid=set())
        path_1 = self.get_random_path(source_0, source_1, points_to_avoid=path_0)

        res = path_0 | path_1
        # print(res)
        return res


        raise Exception("go no further")
        # old way
        # trajectory_0 = [source_0]
        # trajectory_1 = [source_1]
        # display_arr = make_none_array((self.x_size, self.y_size))  # for debugging
        # display_arr = display_arr.astype(str)
        # display_arr[x, y] = "S"
        # LEFT = 0
        # RIGHT = -1
        # 
        # while True:
        #     # two starting points, each has two paths coming away from it
        #     # the paths from the same starting point are not allowed to intersect each other or themselves
        #     # when paths from different sources meet, cut off the paths and take the trajectories up to that point
        #     neighbors_0_left  = self.get_neighbors(*trajectory_0[0])
        #     neighbors_0_right = self.get_neighbors(*trajectory_0[-1])
        #     neighbors_1_left  = self.get_neighbors(*trajectory_1[0])
        #     neighbors_1_right = self.get_neighbors(*trajectory_1[-1])
        #     coords_0_left  = random.choice([p for p in neighbors_0_left if p not in trajectory_0])
        #     coords_0_right = random.choice([p for p in neighbors_0_right if p not in trajectory_0])
        #     coords_1_left  = random.choice([p for p in neighbors_1_left if p not in trajectory_1])
        #     coords_1_right = random.choice([p for p in neighbors_1_right if p not in trajectory_1])
        # 
        #     if coords_0_left in trajectory_1:
        #         cut_point = coords_0_left
        #         trajectory_0 = [coords_0_left] + trajectory_0
        #         # display_arr[coords_0] = "0"
        #         break
        #     elif coords_1_left in trajectory_0:
        #         cut_point = coords_1_left
        #         trajectory_1 = [coords_1_left] + trajectory_1
        #         # display_arr[coords_1] = "1"
        #         break
        # 
        #     trajectory_0 = [coords_0_left] + trajectory_0 + [coords_0_right]
        #     trajectory_1 = [coords_1_left] + trajectory_1 + [coords_1_right]
        #     # display_arr[coords_0] = "0"
        #     # display_arr[coords_1] = "1"
        #     # print(display_arr)
        #     # input("debug")
        # 
        # cut_index_0 = trajectory_0.index(cut_point)
        # cut_index_1 = trajectory_1.index(cut_point)
        # source_index_0 = trajectory_0.index(source_0)
        # source_index_1 = trajectory_1.index(source_1)
        # begin_0 = min(cut_index_0, source_index_0)
        # end_0 = max(cut_index_0, source_index_0)
        # begin_1 = min(cut_index_1, source_index_1)
        # end_1 = max(cut_index_1, source_index_1)
        # trajectory_0_cut = trajectory_0[begin_0 : end_0]
        # trajectory_1_cut = trajectory_1[begin_1 : end_1]
        # points = set(trajectory_0_cut) | set(trajectory_1_cut) | {cut_point}
        # for p in points:
        #     self.fill_position(p[0], p[1], 0)

    def add_random_zero_loop(self):
        points = self.get_random_zero_loop()
        for p in points:
            self.fill_position(p[0], p[1], 0)

    def fill_elevations_outward_propagation(self):
        # old
        raise Exception("do not use")
        # try filling the neighbors in "shells", get all the neighbors at one step and do all of them first before moving on
        # but that leads to a square propagating outward, so should sometimes randomly restart the list
        while len(self.unfilled_neighbors) + len(self.unfilled_inaccessible) > 0:
            to_fill_this_round = [x for x in self.unfilled_neighbors]
            random.shuffle(to_fill_this_round)
            for p in to_fill_this_round:
                # p = random.choice(list(self.unfilled_neighbors))
                # neighbor = random.choice([x for x in self.get_neighbors(*p) if x in self.filled_positions])
                # neighbor_el = self.get_value_at_position(*neighbor)
                filled_neighbors = [x for x in self.get_neighbors(*p) if x in self.filled_positions]
                average_neighbor_el = np.mean([self.get_value_at_position(*n) for n in filled_neighbors])
                d_el = np.random.normal(0, 100)
                el = average_neighbor_el + d_el
                self.fill_position(p[0], p[1], el)
                if random.random() < 0.02:
                    break  # for
                    # then will have new list of neighbors
            self.draw()

    def fill_elevations(self, n_steps, expected_change_size, plot_every_n_steps=None):
        if plot_every_n_steps is not None:
            plt.ion()
        i = 0
        t0 = datetime.now()
        while True:
            if n_steps is None:
                raise Exception("do not do this anymore, just run until there is sufficient convergence")
                # if len(self.untouched_points) == 0:
                #     break
            else:
                if i >= n_steps:
                    break
            if i % 100 == 0:
                try:
                    dt = datetime.now() - t0
                    n_left = n_steps - i
                    secs_per_step = dt/i
                    eta = secs_per_step * n_left
                    eta_str = str(eta)
                    print("step {}, {} elapsed, {} ETA".format(i, dt, eta_str))
                except ZeroDivisionError:
                    print("div/0!")
            self.make_random_elevation_change(expected_change_size, positive_feedback=True)
            # print("now have {} untouched points".format(len(self.untouched_points)))
            if plot_every_n_steps is not None and i % plot_every_n_steps == 0:
                try:
                    self.draw()
                except ValueError:
                    print("skipping ValueError")
                # input("debug")
            i += 1

    def plot(self):
        plt.gcf()
        self.pre_plot()
        plt.show()

    def draw(self):
        plt.gcf().clear()
        self.pre_plot()
        plt.draw()
        plt.pause(0.001)

    def save_plot_image(self, output_fp):
        self.pre_plot()
        plt.savefig(output_fp)

    def pre_plot(self, alpha=1):
        # plt.imshow(self.array, interpolation="gaussian")  # History.py uses contourf rather than imshow
        min_elevation = self.array.min()
        max_elevation = self.array.max()
        n_sea_contours = 20
        n_land_contours = 100
        if min_elevation < 0:
            sea_contour_levels = np.linspace(min_elevation, 0, n_sea_contours)
        else:
            sea_contour_levels = [0]
        if max_elevation > 0:
            land_contour_levels = np.linspace(0, max_elevation, n_land_contours)
        else:
            land_contour_levels = [0]
        assert sea_contour_levels[-1] == land_contour_levels[0] == 0
        contour_levels = list(sea_contour_levels[:-1]) + list(land_contour_levels)
        colormap = get_land_and_sea_colormap()
        # care more about seeing detail in land contour; displaying deep sea doesn't matter much
        max_color_value = max_elevation
        min_color_value = -1 * max_elevation

        # draw colored filled contours
        plt.contourf(self.array, cmap=colormap, levels=contour_levels, vmin=min_color_value, vmax=max_color_value, alpha=alpha)
        try:
            plt.colorbar()
        except IndexError:
            # np being stupid when there are too few contours
            pass

        # draw contour lines, maybe just one at sea level
        plt.contour(self.array, levels=[min_elevation, 0, max_elevation], colors="k")

        plt.gca().invert_yaxis()
        # max_grad, pair = self.get_max_gradient()
        # p, q = pair
        # print("max gradient is {} from {} to {}".format(max_grad, p, q))

    def plot_gradient(self):
        varray = self.array
        vgrad = np.gradient(varray)
        grad_mag = np.sqrt(vgrad[0]**2 + vgrad[1]**2)
        grad_angle = np.angle(vgrad[0] + 1j*vgrad[1])
        angle_colormap = plt.cm.hsv  # something cyclic
        angle_color = angle_colormap(grad_angle)
        plt.imshow(grad_angle, cmap=angle_colormap, vmin=-np.pi, vmax=np.pi)
        plt.colorbar()
        plt.show()

    def create_rainfall_array(self):
        if hasattr(self, "rainfall_array") and self.rainfall_array is not None:
            return
        self.rainfall_array = np.random.uniform(-100, 100, size=self.array.shape)
        # negative values correspond to more evaporation than rain
        # treat units as height units per tick of time, for flow simulation

    def create_flow_array(self):
        # div(water_flow) is zero everywhere, whether it leaves by flowing or evaporating or whatever
        # so water flow array should tell what the volumetric flow *through* the point is
        self.create_rainfall_array()
        flow_array = np.zeros((self.x_size, self.y_size))
        flow_destination_array = np.full((self.x_size, self.y_size), None)
        # treat sea level as fixed, water flow into and out of elevations below 0 is ignored
        points_sorted_by_decreasing_elevation = []
        for x in self.x_range:
            for y in self.y_range:
                if self.is_land(x, y):
                    elevation = self.array[x, y]
                    tup = (elevation, x, y)
                    points_sorted_by_decreasing_elevation.append(tup)
        points_sorted_by_decreasing_elevation = sorted(points_sorted_by_decreasing_elevation, reverse=True)
        gx, gy = np.gradient(-1*self.array)
        downhill_neighbor_offset = {
            -180: (-1, 0),
            -135: (-1, -1),
            -90:  (0, -1),
            -45:  (1, -1),
            0:    (1, 0),
            45:   (1, 1),
            90:   (0, 1),
            135:  (-1, 1),
            180:  (-1, 0),
        }
        for el, x, y in points_sorted_by_decreasing_elevation:
            dx = gx[x, y]
            dy = gy[x, y]
            grad_angle = np.angle(dx + 1j*dy, deg=True)
            # print("dx {} dy {} angle {} deg".format(dx, dy, grad_angle))
            rounded_to_45_deg = int(45*round(grad_angle/45))
            # input("rounded to {}".format(rounded_to_45_deg))
            downhill_x_offset, downhill_y_offset = downhill_neighbor_offset[rounded_to_45_deg]
            downhill_neighbor = (x + downhill_x_offset, y + downhill_y_offset)
            nx, ny = downhill_neighbor
            flow_array[x, y] += self.rainfall_array[x, y]
            if self.is_valid_point(nx, ny):
                flow_array[nx, ny] += flow_array[x, y]
                flow_destination_array[x, y] = (nx, ny)
            # else:
            #     input("invalid neighbor {}".format((nx, ny)))
        self.flow_array = flow_array
        self.flow_destination_array = flow_destination_array

        qs = np.linspace(0, 1, 100)
        flow_array_no_zeros = flow_array[flow_array != 0]
        quantiles = {q: np.quantile(flow_array_no_zeros, q) for q in qs}
        def get_nearest_quantile(x):
            if x < quantiles[qs[0]]:
                return 0
            if x > quantiles[qs[-1]]:
                return 1
            for q0, q1 in zip(qs[:-1], qs[1:]):
                v0 = quantiles[q0]
                v1 = quantiles[q1]
                # print(v0, x, v1)
                if v0 <= x <= v1:
                    # input("match")
                    if abs(q0-x) > abs(q1-x):
                        return q0
                    else:
                        return q1
            raise RuntimeError("no quantile found for value {}".format(x))

        flow_quantile_array = np.zeros((self.x_size, self.y_size))
        for x in self.x_range:
            for y in self.y_range:
                flow_quantile_array[x, y] = get_nearest_quantile(flow_array[x, y])
        self.flow_quantile_array = flow_quantile_array
        
    def is_land(self, x, y):
        # TODO: make it possible for land to be below sea level
        return self.array[x, y] >= 0

    def plot_flow_amounts(self):
        self.create_flow_array()
        # arr = self.flow_array  # max is too high for linear cmap
        arr = self.flow_quantile_array
        plt.imshow(arr, cmap=plt.cm.inferno)
        plt.colorbar()
        plt.show()

    def plot_rivers(self):
        self.pre_plot(alpha=1)
        self.create_flow_array()
        print("flow stats: min {} median {} mean {} max {}".format(
            np.min(self.flow_array),
            np.median(self.flow_array),
            np.mean(self.flow_array),
            np.max(self.flow_array),
        ))
        mean_flow = np.mean(self.flow_array)

        # blue_value = lambda x: get_nearest_quantile(x)
        # alpha_value = lambda x: get_nearest_quantile(x)
        # river_rgba_array = []
        line_segments = []
        colors = []
        for x in self.x_range:
            # this_row = []
            for y in self.y_range:
                flow = self.flow_array[x, y]
                flow_destination = self.flow_destination_array[x, y]
                flow_quantile = self.flow_quantile_array[x, y]
                if flow_destination is not None:
                    # seg = [(x, y), flow_destination]
                    # transpose
                    seg = [(y, x), flow_destination[::-1]]
                    # print(seg)
                    line_segments.append(seg)
                    # b = blue_value(flow)
                    # a = alpha_value(flow)
                    # r = 0
                    # g = b  # make it cyan
                    # color = (r, g, b, a)
                    # color = (0, 0, 1, a)
                    # if a > 0:
                    #     input("flow {} from {} to {} gave rgba {}".format(flow, (x, y), flow_destination, color))
                    cmap = plt.cm.GnBu
                    r, g, b, a = cmap(flow_quantile)
                    color = (r, g, b, a*0.5)
                    colors.append(color)
                # this_row.append(color)
            # river_rgba_array.append(this_row)
        # river_rgba_array = np.array(river_rgba_array)
        # print(test_array)
        # print(f(test_array))
        # print(river_rgba_array[150:152, 150:153])
        # plt.imshow(self.water_flow_array)
        # plt.imshow(river_rgba_array, origin="lower")
        lc = mcollections.LineCollection(line_segments, colors=colors)
        plt.gca().add_collection(lc)
        plt.gca().autoscale()
        # n_segments = len(line_segments)
        # i = 0
        # for seg, color in zip(line_segments, colors):
        #     if i % 100 == 1:
        #         print(i, n_segments)
        #     i += 1
        #     p0, p1 = seg
        #     plt.plot(p0, p1, c=color)
        plt.show()

        # https://stackoverflow.com/questions/16529892/adding-water-flow-arrows-to-matplotlib-contour-plot
        # self.pre_plot()
        # xi, yi = np.meshgrid(self.x_range, self.y_range)
        # zi = self.array
        # dy, dx = np.gradient(-zi.T)
        # print(xi.shape, yi.shape, dx.shape, dy.shape, zi.shape)
        # plt.streamplot(self.x_range, self.y_range, dx, dy, color='0.8', density=2)
        # contours = plt.contour(xi, yi, zi, linewidths=2)
        # plt.clabel(contours)
        # plt.show()

    @staticmethod
    def from_image(image_fp, color_condition_dict, default_color):
        # all points in image matching something in the color dict should be that color no matter what
        # everything else is randomly generated
        # i.e., put the determined points in points_to_avoid for functions that take it
        if any(len(x) != 4 for x in color_condition_dict.keys()):
            raise ValueError("all color keys must have length 4, RGBA:\n{}".format(color_condition_dict.keys()))
        im = Image.open(image_fp)
        width, height = im.size
        m = Map(height, width)  # rows, columns
        arr = np.array(im)
        color_and_first_seen = {}
        for x in range(m.x_size):
            for y in range(m.y_size):
                color = tuple(arr[x, y])
                if color not in color_condition_dict:
                    color = default_color
                    arr[x, y] = color
                if color not in color_and_first_seen:
                    print("got new color {} at {}".format(color, (x, y)))
                    color_and_first_seen[color] = (x, y)
                if color in color_condition_dict:
                    fill_value, condition, is_frozen = color_condition_dict[color]
                    if fill_value is None:
                        fill_value = 0
                    m.fill_position(x, y, fill_value)
                    m.add_condition_at_position(x, y, condition)
                    if is_frozen:
                        m.freeze_point(x, y)
        return m

    @staticmethod
    def load_elevation_data(data_fp):
        with open(data_fp) as f:
            lines = f.readlines()
        lines = [[float(x) for x in line.split(",")] for line in lines]
        array = np.array(lines)
        x_size, y_size = array.shape
        return Map(x_size, y_size, array=array)

    def freeze_coastlines(self):
        coastal_points = set()
        for x in range(self.x_size):
            for y in range(self.y_size):
                if self.array[x, y] < 0:
                    neighbors = self.get_neighbors(x, y)
                    for n in neighbors:
                        if self.array[n[0], n[1]] >= 0:
                            coastal_points.add(n)
        for p in coastal_points:
            self.fill_position(p[0], p[1], 0)
            self.freeze_point(*p)

    # def get_min_gradient_array(self):
    #     if hasattr(self, "min_gradient_array") and self.min_gradient_array is not None:
    #         return self.min_gradient_array
    #     res = make_nan_array(self.x_size, self.y_size)
    #     for p in self.get_all_points():
    #         min_grad_this_point = np.inf
    #         for q in self.get_neighbors(*p):
    #             dist = 1 if p[0] == q[0] or p[1] == q[1] else np.sqrt(2)
    #             dh = self.array[q[0], q[1]] - self.array[p[0], p[1]]
    #             grad = dh/dist
    #             min_grad_this_point = min(grad, min_grad_this_point)
    #         res[p[0], p[1]] = min_grad_this_point
    #     self.min_gradient_array = res
    #     return res

    def get_max_gradient(self):
        print("getting max grad")
        from_point = None
        to_point = None
        max_grad = -np.inf
        max_grad_pair = None
        all_points = sorted(self.get_all_points())
        for p in all_points:
            for q in self.get_neighbors(*p):
                dist = 1 if p[0] == q[0] or p[1] == q[1] else np.sqrt(2)
                dh = self.array[q[0], q[1]] - self.array[p[0], p[1]]
                grad = dh/dist
                if grad > max_grad:
                    max_grad = grad
                    max_grad_pair = (p, q)
        return max_grad, max_grad_pair

    def save_elevation_data(self, output_fp):
        # format is just grid of comma-separated numbers
        # if not confirm_overwrite_file:
        #     return
        open(output_fp, "w").close()  # clear file
        for x in range(self.x_size):
            this_row = ""
            for y in range(self.y_size):
                el = self.array[x, y]
                this_row += "{:.1f},".format(el)
            assert this_row[-1] == ","
            this_row = this_row[:-1] + "\n"
            open(output_fp, "a").write(this_row)
        print("finished saving elevation data to {}".format(output_fp))


def confirm_overwrite_file(output_fp):
    if os.path.exists(output_fp):
        yn = input("Warning! Overwriting file {}\ncontinue? (y/n, default n)".format(output_fp))
        if yn != "y":
            print("aborting")
            return False
    return True

def defect():
    # sea becomes land or vice versa
    return random.random() < 0.05



if __name__ == "__main__":
    # show_elevation_change_functions()
    from_image = False
    from_data = True
    image_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/WorldMapScanPNGs/"
    if from_image:
        # image_fp_no_dir = "LegronCombinedDigitization_ThinnedBorders_Final.png"
        # image_fp_no_dir = "MientaDigitization_ThinnedBorders_Final.png"
        # image_fp_no_dir = "TestMap3_ThinnedBorders.png"
        # image_fp_no_dir = "TestMap_NorthernMystIslands.png"
        # image_fp_no_dir = "TestMap_Jhorju.png"
        # image_fp_no_dir = "TestMap_Amphoto.png"
        # image_fp_no_dir = "TestMap_Mako.png"
        # image_fp_no_dir = "TestMap_Myst.png"
        image_fp_no_dir = "TestMap_Ilausa.png"
        # image_fp_no_dir = "TestMap_VerticalStripes.png"
        # image_fp_no_dir = "TestMap_AllLand.png"
        # image_fp_no_dir = "TestMap_CircleIsland.png"
        image_fp = image_dir + image_fp_no_dir

        elevation_data_output_fp = image_dir + "ElevationGenerationOutputData_" + image_fp_no_dir.replace(".png", ".txt")
        plot_image_output_fp = image_dir + "ElevationGenerationOutputPlot_" + image_fp_no_dir
    
        color_condition_dict = {
            # (  0,  38, 255, 255): (0,  lambda x: x == 0, True),  # dark blue = sea level
            (  0, 255, 255, 255): (-1, lambda x: x < 0, False),  # cyan = sea
            (  0,   0,   0, 255): (1, lambda x: x > 0, False),
            # (  0, 255,  33, 255): (1,  lambda x: x > 0 or defect(), False),  # green = land
            # (255,   0,   0, 255): (1,  lambda x: x > 0 or defect(), False),  # red = land (country borders)
        }
        default_color = (0, 0, 0, 255)
        m = Map.from_image(image_fp, color_condition_dict, default_color)
        m.freeze_coastlines()
        generate_elevation_changes = True
    elif from_data:
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_CircleIsland.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_LegronCombinedDigitization_ThinnedBorders_Final.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_MientaDigitization_ThinnedBorders_Final.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Mako.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Amphoto.txt"
        data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Jhorju.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_Ilausa.txt"
        # data_fp_no_dir = "ElevationGenerationOutputData_TestMap_NorthernMystIslands.txt"
        data_fp = image_dir + data_fp_no_dir
        m = Map.load_elevation_data(data_fp)
        generate_elevation_changes = False
    else:
        m = Map(300, 500)
        m.fill_all(0)
        elevation_data_output_fp = "/home/wesley/programming/ElevationGenerationOutputData_Random.png"
        plot_image_output_fp = "/home/wesley/programming/ElevationGenerationOutputPlot_Random.png"
        generate_elevation_changes = True
        
    print("map size {} pixels".format(m.size()))

    if generate_elevation_changes:
        expected_change_size = 10000
        expected_touches_per_point = 200
        n_steps = int(expected_touches_per_point / expected_change_size * m.size())
        # n_steps = np.inf
        # n_steps = 10000
        plot_every_n_steps = None
        print("filling elevation for {} steps, plotting every {}".format(n_steps, plot_every_n_steps))
        m.fill_elevations(n_steps, expected_change_size, plot_every_n_steps)
        # m.plot()
        m.save_elevation_data(elevation_data_output_fp)
        m.save_plot_image(plot_image_output_fp)
    else:
        # m.plot()
        # m.plot_gradient()
        m.plot_flow_amounts()
        m.plot_rivers()
