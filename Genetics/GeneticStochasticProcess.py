# idk what my goal is here, just play around with genetic algorithm stuff
# maybe use it to generate a stochastic process

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from Music.WavUtil import write_signal_to_wav


def get_random_genome(length):
    return "".join(random.choice("01") for x in range(length))


def get_function_constituents(g):
    constituents = []
    # iterate through the genome as a loop, until reach a "stop" message
    # give an absolute time limit for the genome to return, or kill the organism
    time_limit = 0.1
    t0 = time.time()
    while time.time() - t0 < time_limit:
        # will add a constituent function of form c*f(m*t+b)
        simple_function_codon = g[0:3]
        outer_coefficient_codon = g[3:6]
        inner_coefficient_codon = g[6:9]
        phase_codon = g[9:12]
        
        simple_function = get_simple_function(simple_function_codon)
        if simple_function is None:  # received stop message
            return constituents
        outer_coefficient = get_number(outer_coefficient_codon)
        inner_coefficient = get_number(inner_coefficient_codon)
        phase = get_number(phase_codon)
        d = {"f": simple_function, "c": outer_coefficient, "m": inner_coefficient, "b": phase}
        constituents.append(d)

        # rotate the genome
        g = g[12:] + g[:12]
    return None


def get_simple_function(codon):
    simple_functions = {
        "000": lambda x: x,
        "001": lambda x: np.sin(x),
        "010": lambda x: np.cos(x),
        "011": lambda x: np.tan(x),
        "100": lambda x: 1.0/x,
        "101": lambda x: x**2,
        "110": lambda x: abs(x)**0.5,
        "111": None,  # stop
    }
    return simple_functions[codon]


def get_number(codon):
    # first bit is sign, others are binary decimals, all results will have abs < 1
    sign = 1 if codon[0] == "1" else -1 if codon[0] == "0" else None
    res = 0
    for i in range(1, len(codon)):
        res += int(codon[i]) * 2**-i
    res *= sign
    return res


def get_trajectory(g, initial_condition):
    constituents = get_function_constituents(g)
    if constituents is None:
        # genome did not terminate, kill the organism
        return None
    # print("{} constituents".format(len(constituents)))
    x = initial_condition
    t_max = 1000
    xs = [x]
    for t in range(t_max):
        try:
            constituent_values = [d["c"]*d["f"](d["m"]*x+d["b"]) for d in constituents]
            x = sum(constituent_values)
            xs.append(x)
        except (ValueError, ZeroDivisionError, OverflowError):
            # kill the organism
            return None
    return xs


def select_environmental_survivors(organisms):
    survivors = {}
    initial_condition = random.random()
    for g in organisms:
        # print(g)
        # print(initial_condition)
        xs = get_trajectory(g, initial_condition)
        if xs is None:
            # organism was killed
            continue
        # plt.plot(xs)
        # plt.show()
        if trajectory_survives(xs):
            score = score_trajectory(xs)
            survivors[g] = score
    return survivors


def trajectory_survives(xs):
    # minimum requirements for survival, will be further culled based on score later
    min_val = -1000
    max_val = 1000
    return all(min_val <= x <= max_val for x in xs) and np.std(xs) > 0


def score_trajectory(xs):
    # use component simpler scoring functions, can play with their weights
    components = []
    components.append(proportion_small_diffs_score(xs, 1) * -1000)
    components.append(autocorr(xs, 1) * -100)
    components.append(autocorr(xs, 2) * -100)
    components.append(autocorr(xs, 3) * -100)
    components.append(autocorr(xs, 5) * -100)
    components.append(autocorr(xs, 7) * -100)
    components.append(autocorr(xs, 11) * -100)
    components.append(median_abs_diff_score(xs, 1) * 1)
    components.append(median_abs_diff_score(xs, 2) * 1)
    components.append(median_abs_diff_score(xs, 3) * 1)
    components.append(median_abs_diff_score(xs, 5) * 1)
    components.append(median_abs_diff_score(xs, 7) * 1)
    components.append(median_abs_diff_score(xs, 11) * 1)

    return sum(components)


def median_abs_diff_score(xs, n):
    # reward lots of movement, less boring processes
    dxs = np.diff(xs, n)
    return np.median(abs(dxs))


def sign_diff_score(xs, n_x_diff, n_sign_diff):
    # higher score for the sign switching, i.e., oscillation in the trajectory
    dxs = np.diff(xs, n_x_diff)
    signs = np.sign(dxs)
    d_signs = np.diff(signs, n_sign_diff)
    return np.mean(abs(d_signs))


def proportion_small_diffs_score(xs, n):
    dxs = np.diff(xs, n)
    full_range = max(xs) - min(xs)
    small_diff_proportion = 0.01
    small_diff = small_diff_proportion * full_range
    n_small_diffs = (abs(dxs) <= small_diff).sum()
    return n_small_diffs / len(xs)


def autocorr(xs, n_diff):
    a = xs[:-n_diff]
    b = xs[n_diff:]
    return corr(a, b)


def corr(a, b):
    r, p_value = pearsonr(a, b)
    if np.isnan(r):
        return 1  # treat it as fully correlated with itself; NaN happens if the series doesn't move
    return r


def mutate(g):
    # flip bits at random, but only do it rarely
    while random.random() < 0.1:
        i = random.choice(range(len(g)))
        other_bit = {"0":"1", "1":"0"}[g[i]]
        g = g[:i] + other_bit + g[i+1:]
    return g


def get_next_generation(organisms):
    # first reproduce, then cull
    print("{} organisms".format(len(organisms)))
    all_this_generation = []
    for g in organisms:
        mate = random.choice(organisms)
        cut_point_0 = random.randrange(len(g))
        cut_point_1 = cut_point_0 + random.randint(-2,2)
        cut_point_1 = max(min(cut_point_1, len(mate)), 0)
        # put both of these children in there
        child_a = g[:cut_point_0] + mate[cut_point_1:]
        child_b = mate[:cut_point_1] + g[cut_point_0:]
        child_a = mutate(child_a)
        child_b = mutate(child_b)
        all_this_generation.append(g)
        all_this_generation.append(child_a)
        all_this_generation.append(child_b)
    print("{} after reproduction".format(len(all_this_generation)))

    # # stop overpopulation, create infant mortality
    # expected_population = 1000
    # survival_rate = expected_population / len(res)
    # res = [g for g in res if random.random() < survival_rate]
    # return res

    survivors_trajectories = select_environmental_survivors(all_this_generation)
    print("{} survivors".format(len(survivors_trajectories)))

    # now cull survivors based on score (run them through survive function first, though, so no zero scores make it through)
    n_to_keep = 100
    survivors_trajectories = sorted(survivors_trajectories.items(), key=lambda kv: kv[1], reverse=True)
    survivors = [st[0] for st in survivors_trajectories]
    # print("surv", survivors)
    print("selecting top {}".format(n_to_keep))
    survivors = survivors[:n_to_keep]
    lens = np.array([len(g) for g in survivors])
    print("genome length stats: min {}; max {}; avg {}".format(lens.min(), lens.max(), lens.mean()))
    return survivors


def plot_genes(organisms):
    max_len = max(len(g) for g in organisms)
    # pad them with None and turn them into ints
    padded_organisms = []
    for g in organisms:
        padded = [int(c) for c in g] + [np.nan]*(max_len - len(g))
        padded_organisms.append(padded)
    arr = np.array(padded_organisms)
    aspect_ratio = max_len / len(padded_organisms)  # ratio of > 1 will squish x and stretch y axis
    plt.imshow(arr, aspect=aspect_ratio)
    plt.show()


def plot_trajectories(organisms):
    for i in range(10):
        print("plot {}/10".format(i+1))
        plt.subplot(2, 5, i+1)
        initial_condition = 0.1*i + 0.05
        for g in organisms:
            xs = get_trajectory(g, initial_condition)
            plt.plot(xs, alpha=0.1)
    plt.show()


def plot_sample_gallery(organisms):
    if type(organisms) is dict:
        organisms = list(organisms.keys())
    rows = 3
    columns = 6
    n_panels = rows * columns
    for i in range(n_panels):
        organism_number = random.choice(range(len(organisms)))
        g = organisms[organism_number]
        initial_condition = random.random()
        xs = get_trajectory(g, initial_condition)
        plt.subplot(rows, columns, i+1)
        plt.title("org {}: {:.3f}".format(organism_number, initial_condition))
        plt.plot(xs)
    plt.show()


def convert_trajectories_to_wav(trajectories):
    signal_array = []
    framerate = 1000
    wav_rate = 44100
    # will interpolate so the oscillations in the function are not occurring at like 20kHz or something earsplitting
    separator_secs = 0
    separator = [0] * int(wav_rate * separator_secs)  # some number of seconds of silence
    for xs in trajectories:
        xs = np.array(xs)
        max_amp = max(abs(xs))
        xs = xs / max_amp  # normalize to 1=max
        ts = np.arange(len(xs))
        interp_func = interp1d(ts, xs, kind="cubic")
        wav_sample_step = framerate/wav_rate  # how many t steps to go for each point in wav sample (e.g. framerate of 100 from function gives wav sample step of 1/441 second)
        new_ts = np.arange(min(ts), max(ts)+wav_sample_step, wav_sample_step)
        new_xs = interp_func(new_ts) 
        signal_array += list(new_xs) + list(separator)
    write_signal_to_wav(signal_array, "GeneticStochasticProcessOutput.wav")


def create_sample_wav(organisms):
    if type(organisms) is dict:
        organisms = list(organisms.keys())
    trajectories = []
    for i in range(20):
        organism_number = random.choice(range(len(organisms)))
        g = organisms[organism_number]
        initial_condition = random.random()
        xs = get_trajectory(g, initial_condition)
        trajectories.append(xs)
    convert_trajectories_to_wav(trajectories)


if __name__ == "__main__":
    organisms = [get_random_genome(100) for i in range(10)]
    for generation in range(25):
        print("generation {}".format(generation))
        organisms = get_next_generation(organisms)

    # keep those that still survive, so plotting trajectories will behave
    organisms = select_environmental_survivors(organisms)

    for g in sorted(organisms):
        print(g)

    plot_genes(sorted(organisms))

    # plot_trajectories(organisms)
    plot_sample_gallery(organisms)
    create_sample_wav(organisms)
