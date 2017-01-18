import itertools
import math
import random
import time
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


def get_square_matrix(mu, sigma, n):
    return np.random.normal(mu, sigma, (n, n))


def get_mutation_matrix(n):
    return np.exp(get_square_matrix(0, 0.01, n))


def mutate(matrix):
    n = matrix.shape[0]
    assert matrix.shape[1] == n
    mutation = get_mutation_matrix(n)
    return mutation * matrix


def get_upset_matrix(n):
    return np.random.choice([-1 ,1], (n, n))


def upset(matrix):
    n = matrix.shape[0]
    assert matrix.shape[1] == n
    mutation = get_upset_matrix(n)
    return mutation * matrix


def get_organism_matrix(n):
    return get_square_matrix(0, 10, n)


def get_environment_matrix(n):
    return get_square_matrix(0, 1, n)


def get_offspring(organism1, organism2):
    dims = organism1.shape + organism2.shape
    n, *others = dims
    assert len(dims) == 4
    assert all(x == n for x in others)

    alphas = get_square_matrix(0.5, 0.4, n)
    return (organism1 * alphas) + (organism2 * (1 - alphas))


def evaluate(organism, environment):
    m = np.dot(organism, environment)
    return (m.max() - m.min()) / (organism.max() - organism.min())


def sort_organisms(organisms, environment):
    return sorted(organisms, key=lambda x: evaluate(x, environment), reverse=True)


def reproduce(organisms):
    n = organisms[0].shape[0]
    new_organisms = [get_organism_matrix(n) for x in organisms]

    for combo in itertools.combinations(organisms, 2):
        if np.random.rand() < 0.4:
            new_organisms.append(get_offspring(*combo))

    for organism in organisms:
        new_organisms.append(upset(organism))

    return organisms + new_organisms


def kill_off(organisms):
    return [x for x in organisms if random.random() < 0.8]


def run(plot=True):
    n = 9
    n_organisms = 20
    organisms = [get_organism_matrix(n) for i in range(n_organisms)]
    environment = get_environment_matrix(n)

    np.set_printoptions(formatter={'float_kind': lambda x: "{:10.4f}".format(x)})

    organisms = sort_organisms(organisms, environment)

    if plot:
        plt.ion()
        plt.imshow(get_square_matrix(1, 1, 1))
        plt.colorbar()

    iteration = 1
    while True:
        verbose = iteration % 100 == 0
        if verbose:
            if plot:
                plt.clf()
                
                plt.subplot(221)
                plt.imshow(environment, interpolation="none")
                plt.colorbar()
                
                plt.subplot(222)
                plt.imshow(organisms[0], interpolation="none")
                plt.colorbar()
                
                plt.subplot(223)
                plt.imshow(organisms[1], interpolation="none")
                plt.colorbar()

                plt.subplot(224)
                plt.imshow(organisms[-1], interpolation="none")
                plt.colorbar()

                plt.pause(1e-3)
            else:
                print("best organism")
                print(organisms[0])
                print("\nenvironment")
                print(environment)
                print()
                time.sleep(0.1)
            print("best score (iteration {0}): {1}".format(iteration, evaluate(organisms[0], environment)))

        organisms = [mutate(x) for x in organisms]
        organisms = reproduce(organisms)
        organisms = kill_off(organisms)
        organisms = sort_organisms(organisms, environment)[:n_organisms]
        organisms = kill_off(organisms)
        environment = mutate(environment)
        iteration += 1


if __name__ == "__main__":
    run()