import math
import random

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction


def factor(n, starting_prime=2, verbose=False):
    if n > 1e12:
        input("number to be factored is large: {0}; be careful with RAM in the implementation using lists".format(n))
    n = int(n)
    if n == 1:
        return []
    elif n < 1:
        raise ValueError("invalid natural number {0}".format(n))
    sq = int(math.sqrt(n))
    for p in get_primes_between(starting_prime, sq):
        if verbose:
            print("trying prime {0}".format(p))
        if n % p == 0:
            return [p] + factor(n / p, starting_prime=p, verbose=verbose)
    return [n]


def get_factor_counts(n):
    if type(n) is Fraction:
        numer_counts = get_factor_counts(n.numerator)
        denom_counts = get_factor_counts(n.denominator)
        assert set(numer_counts.keys()) & set(denom_counts.keys()) == set(), "fraction does not have coprime numerator and denominator"
        d = {}
        d.update(numer_counts)
        d.update({k: -v for k, v in denom_counts.items()})

    else:
        factors = factor(n)
        d = {}
        for f in factors:
            if f not in d:
                d[f] = factors.count(f)

    return d


def compare_factorization(d1, d2):
    if d1 == d2:
        return 0
    elif less_than_by_factorization(d1, d2):
        return -1
    elif less_than_by_factorization(d2, d1):
        return 1
    else:
        raise Exception("impossible! go check that less_than_by_factorization() works")


def sort_by_factorization(lst, memoized_counts={}):
    # sorting with comparator functions was apparently removed in Python 3
    # optimize later
    
    # randomized quicksort attempt
    if len(lst) <= 1:
        return lst

    memoized_counts.update({x: get_factor_counts(x) for x in lst if x not in memoized_counts})

    lst = random.sample(lst, len(lst))
    pivot = lst[0]
    equals = []
    left = []
    right = []

    for x in lst:
        if x == pivot:
            equals.append(x)
        else:
            dx = memoized_counts[x]
            dp = memoized_counts[pivot]
            assert dx != dp
            if less_than_by_factorization(dx, dp):
                left.append(x)
            elif less_than_by_factorization(dp, dx):
                right.append(x)
            else:
                raise

    return sort_by_factorization(left, memoized_counts) + [pivot] + equals[:-1] + sort_by_factorization(right, memoized_counts)


def less_than_by_factorization(d1, d2):
    # in this ordering, high powers of 2 are the least and large primes are the greatest

    all_keys = set(d1.keys()) | set(d2.keys())
    for p in sorted(all_keys):
        if p not in d1:
            d1[p] = 0
        if p not in d2:
            d2[p] = 0

        c1 = d1[p]
        c2 = d2[p]
        if c1 == c2:
            continue
        return c1 > c2
    raise ValueError("Numbers are equal")


def get_primes_between_ATTEMPT_1(m, n):
    m = int(m)
    n = int(n)
    a = [x for x in range(2, n + 1)]
    b = [True for x in a]

    for i in range(len(a)):
        if b[i]:
            for j in range(i + 1, len(a)):
                if a[j] % a[i] == 0:
                    b[j] = False

    return [a[i] for i in range(len(a)) if b[i] and a[i] >= m]


def get_primes_between_ATTEMPT_2(m, n):
    m = int(m)
    n = int(n)
    a = [x for x in range(2, n + 1)]
    primes = []
    while len(a) > 0:
        q = a[0]
        primes.append(q)
        a = [x for x in a if x % q != 0]
    return [x for x in primes if x >= m]


def get_primes_between(m, n):
    m = int(m)
    n = int(n)
    a = [x for x in range(0, n + 1)]  # FIXME: these lists take up way too much unnecessary memory; can use generator somehow?
    b = [(x >= 2) for x in range(0, n + 1)]
    q = 2
    while q <= n:
        if b[q]:
            if q >= m:
                yield q
            j = 2
            while q * j <= n:
                b[q * j] = False
                j += 1
        q += 1


def get_num_primes_between(m, n):
    return len(get_primes_between(m, n))


def quiz():
    n = random.randint(100, 999)
    factors = factor(n)
    answer = input("Factor {0}.\nEnter prime factors, separated by commas. If a factor occurs more than once, include it multiple times.\n".format(n))
    if sorted([int(x.strip()) for x in answer.split(",")]) == factors:
        print("Correct!")
    else:
        print("Incorrect. The answer is {0}.".format(factors))


def factor_user_input():
    n = input("Number to factor: ")
    print(factor(n, verbose=False))


def tell_num_primes():
    a = input("Number to get primes up to: ")
    print("Number of primes: {0}".format(get_num_primes_between(2, a)))


def show_numer_denom_grid(numers, denoms, sorted_lst):
    positions = {v: i for i, v in enumerate(sorted_lst)}
    grid = np.array([[positions[Fraction(numer, denom)] for denom in denoms] for numer in numers])
    plt.imshow(grid)
    plt.show()


if __name__ == "__main__":
    # while True:
        # quiz()
        # factor_user_input()
        # tell_num_primes()

    numers = [i for i in range(1, 100)]
    denoms = [i for i in range(1, 100)]
    lst = [Fraction(numer, denom) for numer in numers for denom in denoms]  # DON'T use Fraction(float x); the terms will be enormous (e.g. 1.1 == 2476979795053773/2251799813685248)
    sorted_lst = sort_by_factorization(set(lst))

    # show_numer_denom_grid(numers, denoms, sorted_lst)

    plt.plot(sorted_lst, "o")
    plt.show()

    print(["{0}/{1}".format(x.numerator, x.denominator) for x in sorted_lst])