# evolve some bit strings based on slight changes to the environment (i.e. the parameters of the fitness function)

import random
import math
import numpy as np
import matplotlib.pyplot as plt

from RawVariation import get_dna, transcribe_dna, print_dna, plot_dna_as_path, plot_dnas_as_paths
from FunctionOfBits import linear_choice_series, signed_log, plus_minus_cumsum, same_different_direction_path


def get_starting_individuals(n_individuals):
    get_length = lambda: max(10, int(np.random.normal(250,50)))
    return [get_dna(get_length()) for i in range(n_individuals)]


def alpha(a,b,alpha):
    return a + alpha * (b-a)


def get_bidirectional_ratio(a,b):
    # like abs but in log space, so (2,4) and (4,2) will both return 2 instead of 2 vs 1/2
    r = a/b
    log_r = np.log(r)
    abs_log = abs(log_r)
    new_r = np.exp(abs_log)
    return new_r

assert get_bidirectional_ratio(2,4) == get_bidirectional_ratio(6,3) == 2


def run_evolution_in_environment(dnas, eval_func, n_generations):
    birth_factor = 2  # let this be constant, but death factor apprach 0 for large populations (i.e. it decreases them more)
    for gen_i in range(n_generations):
        dnas = reproduce(dnas, birth_factor * np.exp(np.random.normal(0, 0.01)), baseline_error_rate=1/100)
        n = len(dnas)
        death_factor_raw = np.exp(-n/100)
        death_factor = alpha(0.8 * 1/birth_factor, 1, death_factor_raw)  # have it tend to something nonzero for large populations, so we don't kill all individuals for sufficiently large population, but still have the product of birth rate and death rate < 1 for large populations
        death_factor *= np.exp(np.random.normal(0, 0.01))
        dnas = select_survivors(dnas, eval_func, birth_factor, death_factor)
    return dnas


def reproduce(dnas, birth_factor, baseline_error_rate):
    n_individuals_to_create = math.ceil(birth_factor * len(dnas))  # want to round up to compensate for the fact that death rate rounds down
    # just use asexual reproduction, it's easier, but kill off all past-generation individuals so the same one doesn't just monopolize the whole population
    new_individuals = [transcribe_dna(random.choice(dnas), baseline_error_rate) for i in range(n_individuals_to_create)]
    # dna is invalid if it's too short
    new_individuals = [dna for dna in new_individuals if len(dna) > 2]
    return new_individuals  # don't include the old dnas here, want only the offspring to go to next generation


def select_survivors(dnas, eval_func, birth_factor, death_factor):
    # automatically kill any with nonfinite fitness (these are to be considered invalid individuals)
    tups = [(eval_func(dna), dna) for dna in dnas]
    tups = [(v,dna) for v,dna in tups if np.isfinite(v)]  # filter out invalids

    n_individuals_to_kill = int((1-death_factor) * len(dnas))
    assert 0 <= n_individuals_to_kill <= len(dnas), n_individuals_to_kill
    if n_individuals_to_kill == len(dnas):
        return []  # kill everyone
    elif n_individuals_to_kill == 0:
        return dnas  # kill no one

    ascending_score_tups = sorted(tups, key=lambda tup: tup[0])  # I was excited about using lambda tup: (tup[0], *tup[1]) with variable-length tuple unpacking to get around np array comparison problem (.any/.all) but because I'm treating ties such that all dnas with that value have the same fate, sorting by dna at all is not actually necessary here
    kill_cutoff_val = ascending_score_tups[n_individuals_to_kill - 1][0]
    # if this value exists for individuals beyond the number you want to kill, then choose randomly among those with the value
    cutoff_status = {"above": [], "at": [], "below": []}
    for val, dna in tups:
        if val < kill_cutoff_val:
            cutoff_status["below"].append(dna)
        elif val == kill_cutoff_val:
            cutoff_status["at"].append(dna)
        else:
            cutoff_status["above"].append(dna)
    n_at_cutoff_to_kill = n_individuals_to_kill - len(cutoff_status["below"])
    n_at_cutoff_to_keep = len(cutoff_status["at"]) - n_at_cutoff_to_kill
    assert 0 <= n_at_cutoff_to_keep <= len(cutoff_status["at"]), n_at_cutoff_to_keep
    at_cutoff_survivors = random.sample(cutoff_status["at"], n_at_cutoff_to_keep) if len(cutoff_status["at"]) > 0 else []
    survivors = at_cutoff_survivors + cutoff_status["above"]

    n_killed = len(dnas) - len(survivors)
    max_fitness = ascending_score_tups[-1][0]
    min_len = min(len(dna) for dna in survivors)
    max_len = max(len(dna) for dna in survivors)
    print(f"killed {n_killed} of {len(dnas)} individuals; cutoff fitness {kill_cutoff_val}; max fitness {max_fitness}; length range {min_len} - {max_len}")
    return survivors


def run_evolution_in_slightly_different_environments(eval_func, n_params, n_starting_individuals, n_environments, n_generations):
    mean_coefficients = np.random.normal(0, 1, n_params)
    dev = 0  # use dev=0 to see founder effects / chaos with the same starting population in multiple runs of the SAME environment
    environment_deviations = [np.random.normal(0, dev, n_params) for i in range(n_environments)]
    starting_individuals = get_starting_individuals(n_starting_individuals)
    for env_i in range(n_environments):
        print(f"environment #{env_i}")
        coefficients = mean_coefficients + environment_deviations[env_i]
        eval_func_in_env = lambda dna, coefficients=coefficients: eval_func(dna, coefficients)  # lambda closure, don't want coefficients changing outside of this scope and then changing what this lambda does, so assign it as default arg and only pass dna
        evolved_individuals = run_evolution_in_environment(starting_individuals, eval_func_in_env, n_generations)
        print(f"{len(evolved_individuals)} individuals exist after evolution")
        plot_dnas_as_paths(evolved_individuals, save=False)
        plt.savefig(f"EvolvedDnasEnv{env_i}.png")
        plt.gcf().clear()


def autocorrelation(arr, lag):
    if lag >= len(arr) - 1:
        # there will be no overlap or only one point (so correlation is invalid)
        return np.nan
    arr0 = arr[:-lag]
    arr1 = arr[lag:]
    return np.corrcoef(arr0, arr1)[0,1]


def weighted_autocorrelation(arr, coefficients):
    res = 0
    for i, c in enumerate(coefficients):
        autocorr = autocorrelation(arr, lag=i+1)
        res += c * autocorr
    return res


if __name__ == "__main__":
    def eval_func_1(dna, params):
        linear_choice_coefficients = params[:4]
        desired_top_bottom_ratio = np.exp(params[4])
        xs = linear_choice_series(dna, linear_choice_coefficients, modification_function=None)
        xmax = xs.max()
        xmin = xs.min()
        xmean = xs.mean()
        top_half_size = (xmax - xmean)
        bottom_half_size = (xmean - xmin)
        top_bottom_ratio_bidirectional = get_bidirectional_ratio(top_half_size, bottom_half_size)
        # want to reward dna length (so don't just get really short ones) but punish deviation of the top/bottom sizes from each other
        length_reward = (len(dna)) ** (1/4)  # mutation rates should make very long dna maladaptive because it will have offspring with more mutations, so can just reward length itself and treat the mutation stuff as an implicit penalty
        imbalance_penalty = abs(top_bottom_ratio_bidirectional - desired_top_bottom_ratio)
        return length_reward - imbalance_penalty

    def eval_func_2(dna, params):
        autocorrelation_coefficients = abs(params)
        cumsum = plus_minus_cumsum(dna)
        weighted_autocorr = weighted_autocorrelation(cumsum, autocorrelation_coefficients)
        if not np.isfinite(weighted_autocorr):
            return np.nan
        max_weighted_autocorr = sum(autocorrelation_coefficients)
        weighted_autocorr_normalized = weighted_autocorr / max_weighted_autocorr
        assert -1 <= weighted_autocorr_normalized <= 1, (weighted_autocorr_normalized, weighted_autocorr, params, max_weighted_autocorr)
        # reward length, punish lots of weighted autocorrelation (which will be higher for positive autocorr coefficients (autocorrelation at those lags adds to the cost) and lower for negative autocorr coefficients (so those coefficients will end up encouraging survival of dnas with cumsum autocorrelation at those lags))
        autocorr_penalty_factor = 1 - weighted_autocorr_normalized
        return len(dna)**(0.8) * autocorr_penalty_factor**0.5

    def eval_func_1_2(dna, params):
        # this should induce competing goals because f1 likes long straight lines (i.e. most of the dna is either 1 or 0)
        # whereas f2 likes well-bounded cumsums (lots of alternation of bits)
        v1 = eval_func_1(dna, params)
        v2 = eval_func_2(dna, params)
        # print(v1, v2)  # just to make sure the magnitudes are similar enough
        return v1 + v2


    run_evolution_in_slightly_different_environments(eval_func_1_2, n_params=5, n_starting_individuals=10, n_environments=3, n_generations=10)
