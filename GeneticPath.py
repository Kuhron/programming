# make a shape that is a line bent in various ways
# DNA encodes the path that is taken
# 11 is delimiter, other pairs create a ternary number system
# the ternary numbers encode the change in angle for each step

import random
import numpy as np
import matplotlib.pyplot as plt


def get_random_dna(length):
    return [random.choice([0,1]) for i in range(length)]


def get_pairs_from_dna(s):
    # if trailing chars, ignore them
    res = []
    for i in range(len(s)//2):
        segment = s[2*i : 2*i+2]
        res.append(segment)
    return res


def split_pairs_by_delimiter(pairs):
    res = []
    current_run = []
    for p in pairs:
        if p == [1, 1]:  # delimiter
            res.append(current_run)
            current_run = []
        else:
            current_run.append(p)
    res.append(current_run)
    return res


def convert_runs_to_ternary(runs):
    return [convert_pairs_to_ternary(pairs) for pairs in runs]


def convert_pairs_to_ternary(pairs):
    return [convert_pair_to_ternary(p) for p in pairs]


def convert_pair_to_ternary(pair):
    a,b = pair
    return 2*a + b


def convert_ternary_to_number(ternary):
    # treat it as base-3 0.{input} so it will be in interval [0,1]
    res = 0
    for i, x in enumerate(ternary):
        power = -1 * (i+1)
        place_value = 3**power
        res += x * place_value
    return res


def get_angles_from_dna(s):
    pairs = get_pairs_from_dna(s)
    runs = split_pairs_by_delimiter(pairs)
    ternary = convert_runs_to_ternary(runs)
    nums = [convert_ternary_to_number(t) for t in ternary]
    return get_angles_from_nums(nums)


def get_angles_from_nums(nums):
    raw_angles = [2 * np.pi * x for x in nums]
    # want cumulative, so angle of 0 means keep going whatever direction you were just going
    return np.cumsum(raw_angles)


def get_path_points_from_angles(angles):
    # each step, go one unit
    xs = [0]
    ys = [0]
    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        new_x = xs[-1] + dx
        new_y = ys[-1] + dy
        xs.append(new_x)
        ys.append(new_y)
    return xs, ys


def matrix_transform(xs, ys, m):
    xs2 = []
    ys2 = []
    for x, y in zip(xs, ys):
        x2, y2 = np.matmul(m, [x,y])
        xs2.append(x2)
        ys2.append(y2)
    return xs2, ys2


def plot_path_from_dna(s, modification_matrix=None):
    angles = get_angles_from_dna(s)
    xs, ys = get_path_points_from_angles(angles)
    plt.plot(xs, ys, c="b")
    if modification_matrix is not None:
        mod_xs, mod_ys = matrix_transform(xs, ys, modification_matrix)
        plt.plot(mod_xs, mod_ys, c="r")
    plt.show()


def get_fitness(s, m):
    angles = get_angles_from_dna(s)
    xs, ys = get_path_points_from_angles(angles)
    mod_xs, mod_ys = matrix_transform(xs, ys, m)
    avg_d = get_average_difference_from_path_to_modified_path(xs, ys, mod_xs, mod_ys)
    return avg_d  # minimizing the distance will lead to very short dna


def get_average_difference_from_path_to_modified_path(xs, ys, mod_xs, mod_ys):
    res = 0
    for x,y,x2,y2 in zip(xs,ys,mod_xs,mod_ys):
        dx = x2-x
        dy = y2-y
        d = (dx**2 + dy**2) ** 0.5
        res += d
    return res


def get_environment(point_array_shape):
    # 2d array of points, at each of which there is a 2d modification matrix
    # point-array-first form so can index it like m = environment[x,y]
    a_field = get_noise(point_array_shape)
    b_field = get_noise(point_array_shape)
    c_field = get_noise(point_array_shape)
    d_field = get_noise(point_array_shape)
    
    plt.subplot(2,2,1)
    plt.imshow(a_field)
    plt.subplot(2,2,2)
    plt.imshow(b_field)
    plt.subplot(2,2,3)
    plt.imshow(c_field)
    plt.subplot(2,2,4)
    plt.imshow(d_field)
    plt.show()

    # putting it into the right shape, probably a better way to do this
    res = np.zeros(point_array_shape + (2,2))
    # print("res.shape", res.shape)
    for x in range(point_array_shape[0]):
        for y in range(point_array_shape[1]):
            res[x,y,0,0] = a_field[x,y]
            res[x,y,0,1] = b_field[x,y]
            res[x,y,1,0] = c_field[x,y]
            res[x,y,1,1] = d_field[x,y]
    # print("res.shape", res.shape)
    return res


def get_noise(shape):
    arr = np.zeros(shape)
    nx, ny = shape
    xs = range(nx)
    ys = range(ny)
    avg_dim = (nx + ny)/2
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    X, Y = np.meshgrid(xs, ys)
    X = X.T  # idk why np does this
    Y = Y.T
    
    for i in range(100):
        px = random.uniform(min_x, max_x)
        py = random.uniform(min_y, max_y)
        DX = X - px
        DY = Y - py
        D = (DX**2 + DY**2) ** 0.5
        in_region = D < random.uniform(1, avg_dim/2)
        dz = np.random.normal(0, 0.1)
        arr[in_region] += dz

    return arr


def reproduce(s0, s1):
    len0 = len(s0)
    len1 = len(s1)
    avg_len = (len0 + len1)/2
    front_half_len = round(np.random.normal(avg_len/2, 1))
    back_half_len = round(np.random.normal(avg_len/2, 1))
    front_half_len = min(front_half_len, len0, len1)  # in case of 100-sigma event?
    back_half_len = min(back_half_len, len0, len1)

    front_half_0 = s0[:front_half_len]
    front_half_1 = s1[:front_half_len]
    front_half = []
    for x0, x1 in zip(front_half_0, front_half_1):
        front_half.append(random.choice([x0, x1]))
    back_half_0 = s0[:back_half_len]
    back_half_1 = s1[:back_half_len]
    back_half = []
    for x0, x1 in zip(back_half_0, back_half_1):
        back_half.append(random.choice([x0, x1]))

    res = front_half + back_half
    return mutate(res)


def mutate(s):
    # just do point mutations since insertion and deletion occurs in reproduction
    res = []
    for x in s:
        if random.random() < 0.01:
            x = 1-x
        res.append(x)
    return res


def get_random_point_in_shape(shape):
    res = []
    for n in shape:
        res.append(np.random.randint(0, n))
    return res


def initialize_populations(point_array_shape):
    # pick starter points, create random dnas located there
    # run fitness, make them more likely to reproduce with higher fitness (no absolute fitness values should be used, just ranking)
    # offspring are placed either in the same place or at a neighboring point (slightly different environment)
    # lifespan is just one iteration
    # eventually the different starting populations will meet and mix, and adapt to their environments
    # so this is like plants, they can't move, but their offspring can go to a slightly different place
    n_starter_points = 5
    starting_dna_len = 20
    d = {}
    for i in range(n_starter_points):
        px, py = get_random_point_in_shape(point_array_shape)
        individuals = [get_random_dna(starting_dna_len) for j in range(20)]
        d[(px, py)] = individuals
    return d


def reproduce_across_environment(location_dict, environment):
    for px, py in location_dict.keys():
        individuals = location_dict[(px, py)]
        m = environment[px, py]
        fitnesses = [get_fitness(dna, m) for dna in individuals]
        print(fitnesses)
        # TODO they choose mates somehow, reproduce, and then die, offspring may stay in same place or go to a neighboring point

if __name__ == "__main__":
    point_array_shape = (75,100)
    environment = get_environment(point_array_shape)  # 2d array of points at each of which there is a 2d modification matrix
    assert environment.shape == point_array_shape + (2,2), environment.shape

    s0 = get_random_dna(10)
    s1 = get_random_dna(10)
    px, py = get_random_point_in_shape(point_array_shape)
    m = environment[px, py]
    assert m.shape == (2,2), m.shape

    plot_path_from_dna(s0, modification_matrix=m)
    print(get_fitness(s0, m))
    print(get_fitness(s1, m))

    location_dict = initialize_populations(point_array_shape)
    reproduce_across_environment(location_dict, environment)

