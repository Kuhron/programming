# make a shape that is a line bent in various ways
# DNA encodes the path that is taken
# 11 is delimiter, other pairs create a ternary number system
# the ternary numbers encode the change in angle for each step

import random
import math
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
    plt.axis("equal")  # aspect ratio
    plt.show()


def plot_paths_from_dnas(dnas, modification_matrix=None):
    alpha = 0.5
    for s in dnas:
        angles = get_angles_from_dna(s)
        xs, ys = get_path_points_from_angles(angles)
        plt.plot(xs, ys, c="b", alpha=alpha)
        if modification_matrix is not None:
            mod_xs, mod_ys = matrix_transform(xs, ys, modification_matrix)
            plt.plot(mod_xs, mod_ys, c="r", alpha=alpha)
    plt.show()


def get_path_points_from_dna(s):
    angles = get_angles_from_dna(s)
    xs, ys = get_path_points_from_angles(angles)
    return xs, ys


def get_fitness_old(s, m):
    xs, ys = get_path_points_from_dna(s)
    mod_xs, mod_ys = matrix_transform(xs, ys, m)

    # avg_d = get_average_difference_from_path_to_modified_path(xs, ys, mod_xs, mod_ys)
    # return avg_d  # minimizing the distance will lead to very short dna

    res = 0
    for i in range(len(xs)):
        sign = 1 if i % 2 == 0 else -1  # try making every other point pair close/far, to get more interesting outcomes
        dx = mod_xs[i] - xs[i]
        dy = mod_ys[i] - ys[i]
        d = (dx**2 + dy**2) ** 0.5
        res += sign * d
    return res


def get_average_difference_from_path_to_modified_path(xs, ys, mod_xs, mod_ys):
    res = 0
    for x,y,x2,y2 in zip(xs,ys,mod_xs,mod_ys):
        dx = x2-x
        dy = y2-y
        d = (dx**2 + dy**2) ** 0.5
        res += d
    return res


def get_environment(point_array_shape, n_steps=100, dz_stdev=0.1, plot=True):
    # 2d array of points, at each of which there is a 2d modification matrix
    # point-array-first form so can index it like m = environment[x,y]
    a_field = get_noise(point_array_shape, n_steps, dz_stdev)
    b_field = get_noise(point_array_shape, n_steps, dz_stdev)
    c_field = get_noise(point_array_shape, n_steps, dz_stdev)
    d_field = get_noise(point_array_shape, n_steps, dz_stdev)

    if plot:    
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


def get_noise(shape, n_steps=100, dz_stdev=0.1):
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
    
    for i in range(n_steps):
        px = random.uniform(min_x, max_x)
        py = random.uniform(min_y, max_y)
        DX = X - px
        DY = Y - py
        D = (DX**2 + DY**2) ** 0.5
        in_region = D < random.uniform(1, avg_dim/2)
        dz = np.random.normal(0, dz_stdev)
        arr[in_region] += dz

    return arr


def change_environment(environment):
    nx, ny, *_ = environment.shape
    d_environment = get_environment((nx, ny), n_steps=1, dz_stdev=0.01, plot=False)
    return environment + d_environment


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
    n_individuals_per_point = 20
    starting_dna_len = 500
    d = {}
    for i in range(n_starter_points):
        px, py = get_random_point_in_shape(point_array_shape)
        individuals = [get_random_dna(starting_dna_len) for j in range(n_individuals_per_point)]
        d[(px, py)] = individuals
    return d


def reproduce_across_environment(location_dict, environment):
    nx, ny, *_ = environment.shape
    new_location_dict = {}
    for px, py in location_dict.keys():
        individuals = location_dict[(px, py)]
        m = environment[px, py]
        fitnesses = [get_fitness(dna, m) for dna in individuals]
        fitnesses_dnas = sorted(list(zip(fitnesses, individuals)), reverse=True)

        # kill lowest portion of them, randomly mate the others that same number of times
        max_population = random.randint(2, 20)  # random culling events
        fitnesses_dnas = fitnesses_dnas[:max_population]  # kill the worse-off half

        # print(fitnesses_dnas)
        reproducing_individuals = [dna for fit,dna in fitnesses_dnas]
        r0 = 2
        n_offspring_to_make = round(r0 * len(reproducing_individuals))

        for offspring_i in range(n_offspring_to_make):
            s0 = random.choice(reproducing_individuals)
            s1 = random.choice(reproducing_individuals)  # possible to breed with self
            offspring = reproduce(s0, s1)
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            new_px = max(0, min(px + dx, nx-1))
            new_py = max(0, min(py + dy, ny-1))
            if (new_px, new_py) not in new_location_dict:
                new_location_dict[(new_px, new_py)] = []
            new_location_dict[(new_px, new_py)].append(offspring)
    return new_location_dict

        
def add_offspring_to_population(dnas, n_offspring_to_make):
    new_dnas = []
    for offspring_i in range(n_offspring_to_make):
        s0 = random.choice(dnas)
        s1 = random.choice(dnas)  # possible to breed with self
        offspring = reproduce(s0, s1)
        new_dnas.append(offspring)
    return dnas + new_dnas


def cull_population(dnas, max_population, fitness_func):
    fitnesses = [fitness_func(dna) for dna in dnas]
    sorted_fitnesses = sorted(fitnesses, reverse=True)
    top_fitnesses = sorted_fitnesses[:max_population]
    # the lowest one in this list is the lowest acceptable fitness
    bound = min(top_fitnesses)
    return [dna for dna, fitness in zip(dnas, fitnesses) if fitness >= bound]


def simulate_evolution_in_static_environment(fitness_func):
    n_starting_individuals = 200
    max_population = 100
    dna_len = 1000
    dnas = [get_random_dna(dna_len) for i in range(n_starting_individuals)]
    n_steps = 100
    for step_i in range(n_steps):
        if step_i % 10 == 0:
            print(f"step {step_i}/{n_steps}")

        n_offspring_to_make = n_starting_individuals
        dnas = add_offspring_to_population(dnas, n_offspring_to_make)
        dnas = cull_population(dnas, max_population, fitness_func)

    return dnas


def simulate_evolution_in_varying_environments():
    point_array_shape = (10,10)
    environment = get_environment(point_array_shape, plot=False)  # 2d array of points at each of which there is a 2d modification matrix
    assert environment.shape == point_array_shape + (2,2), environment.shape

    location_dict = initialize_populations(point_array_shape)

    for generation_i in range(1000):
        location_dict = reproduce_across_environment(location_dict, environment)
        population = sum(len(dnas) for dnas in location_dict.values())
        print(f"generation {generation_i}: population {population}")
        if population == 0:
            break
        environment = change_environment(environment)

    if population > 0:
        lines_to_write = []
        for px in range(point_array_shape[0]):
            for py in range(point_array_shape[1]):
                lines_to_write.append(f"point {px},{py}\n")
                dnas = location_dict.get((px, py))
                if dnas is not None and len(dnas) > 0:
                    dna_lines = []
                    for dna in dnas:
                        dna_lines.append("".join(str(x) for x in dna) + "\n")
                    dna_lines = sorted(dna_lines)
                    lines_to_write += dna_lines
                else:
                    lines_to_write.append("uninhabited\n")
                lines_to_write.append("\n")
        with open("GeneticPathOutput.txt", "w") as f:
            for line in lines_to_write:
                f.write(line)

        while True:
            px, py = get_random_point_in_shape(point_array_shape)
            dnas = location_dict.get((px, py))
            if dnas is not None and len(dnas) > 0:
                print(f"point {px},{py} has {len(dnas)} individuals")
                m = environment[px, py]
                plot_paths_from_dnas(dnas, modification_matrix=m)
            else:
                print(f"point {px},{py} is uninhabited")


def get_offset_distance_variance(dna, offset):
    # the variance in distance between pairs of points that are {offset} time steps apart
    xs, ys = get_path_points_from_dna(dna)
    xs = np.array(xs)
    ys = np.array(ys)
    xs_diff = abs(xs[:-offset] - xs[offset:])
    ys_diff = abs(ys[:-offset] - ys[offset:])
    distances = np.sqrt(xs_diff**2 + ys_diff**2)
    return np.std(distances)


if __name__ == "__main__":
    # ideally dnas are not competing with each other, but just with their environment (i.e. they survive or they don't, and any survivor creates the same number of offspring as any other survivor)
    while True:
        random_dna = get_random_dna(100000)
        plot_path_from_dna(random_dna)

    # fitness_func = lambda dna: get_offset_distance_variance(dna, 12) - get_offset_distance_variance(dna, 6) - get_offset_distance_variance(dna, 9)
    # dnas = simulate_evolution_in_static_environment(fitness_func)
    # plot_paths_from_dnas(dnas)

    # simulate_evolution_in_varying_environments()
