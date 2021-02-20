from sympy import factorint, primefactors
import numpy as np
import matplotlib.pyplot as plt


def number_of_factors_without_multiplicity(n):
    return len(primefactors(n))


def number_of_factors_with_multiplicity(n):
    return sum(factorint(n).values())


def get_even_minus_odd_factors_sequence(n_min, n_max, with_multiplicity=True):
    res = []
    number_of_numbers_with_even_number_of_factors = 0
    number_of_numbers_with_odd_number_of_factors = 0
    for n in range(1, n_max+1):
        if n % 10000 == 0:
            print("{}/{}".format(n, n_max))
        if with_multiplicity:
            n_factors = number_of_factors_with_multiplicity(n)
        else:
            n_factors = number_of_factors_without_multiplicity(n)

        if n_factors % 2 == 0:
            number_of_numbers_with_even_number_of_factors += 1
        else:
            number_of_numbers_with_odd_number_of_factors += 1

        if n >= n_min:
            # don't add to the list if we won't plot values, but still need to calculate from 1 every time
            current_diff = number_of_numbers_with_even_number_of_factors - number_of_numbers_with_odd_number_of_factors
            res.append(current_diff)
    return res


def plot_conjecture(n_min, n_max, with_multiplicity=True):
    diffs = get_even_minus_odd_factors_sequence(n_min, n_max, with_multiplicity)
    plt.plot(diffs, c="b")
    plt.plot([0] * len(diffs), c="r")
    plt.show()



if __name__ == "__main__":
    # failure range: 906,150,257 ≤ n ≤ 906,488,079
    # the conjecture is for multiplicity=True
    n_min = 1
    n_max = 1000000
    with_multiplicity = True
    plot_conjecture(n_min, n_max, with_multiplicity)
