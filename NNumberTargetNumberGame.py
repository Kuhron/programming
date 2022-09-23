# like the card game 24 or 163, where there are four cards and you have to make that number using the cards and elementary operations

import random
import itertools
import math


def target_is_possible(target, numbers):
    # have to use all the cards
    if len(numbers) == 0:
        raise ValueError
    elif len(numbers) == 1:
        return numbers[0] == target
    # depth-first search, since we need to know the results from all cards regardless of what choices we make
    for new_numbers in get_possible_reductions(numbers):
        if target_is_possible(target, new_numbers):
            return True
    return False


def get_possible_results(numbers):
    # can recurse: choose a pair and "collapse" it into a single number using some operation
    # get two indices without regard to order
    if len(numbers) == 0:
        raise ValueError("empty list")
    elif len(numbers) == 1:
        return set(numbers)
    res = set()
    for new_numbers in get_possible_reductions(numbers):
        res |= get_possible_results(new_numbers)
    return res


def get_possible_reductions(numbers):
    # this returns the ways you can turn N cards into N-1 cards by combining some pair with some operation
    indices = list(range(len(numbers)))
    for i,j in itertools.combinations(indices, 2):
        a = numbers[i]
        b = numbers[j]
        # print(f"a = {a}; b = {b}")
        for operation in operations():
            new_number = operation(a, b)
            new_numbers = [n for k,n in enumerate(numbers) if k != i and k != j] + [new_number]
            yield new_numbers


def operations():
    return [plus, a_minus_b, b_minus_a, mult, a_div_b, b_div_a, a_pow_b, b_pow_a]


class SignlessInfinity:
    def __repr__(self):
        return "fin"


inf = float("inf")
nan = float("nan")
fin = SignlessInfinity()
plus = lambda a, b: a + b
a_minus_b = lambda a, b: a - b
b_minus_a = lambda a, b: b - a
mult = lambda a, b: a * b
a_div_b = lambda a, b: div(a, b)
b_div_a = lambda a, b: div(b, a)
div = lambda x, y: x / y if y != 0 else inf if x > 0 else -inf if x < 0 else nan
a_pow_b = lambda a, b: exp(a, b)
b_pow_a = lambda a, b: exp(b, a)


# treat any result over 1 million in magnitude as infinity
def exp(x, y):
    if x == 0:
        if y > 0:
            return 0
        elif y == 0:
            return 1
        else:
            return fin
    log_exp = y * math.log10(abs(x))
    max_log = 6
    if -max_log <= log_exp <= max_log:
        return x ** y
    elif log_exp > max_log:
        if x > 0:
            return inf
        elif x < 0:
            if y % 2 == 0:
                return inf
            else:
                return -inf
        else:
            return nan
    else:
        return 0


def print_results(results):
    real_results = sorted(x for x in results if type(x) in [int, float])
    complex_results = sorted((x for x in results if type(x) is complex), key=lambda z: (z.real, z.imag))
    fin_results = [x for x in results if type(x) is SignlessInfinity]
    sorted_results = real_results + complex_results + fin_results
    assert len(sorted_results) == len(results)
    print(sorted_results)
    print(f"cards {cards} gave {len(results)} results")


def get_all_card_possibilities(n_cards, min_value, max_value):
    return itertools.combinations_with_replacement(list(range(min_value, max_value + 1)), n_cards)


def get_target_occurrence(target, n_cards, min_value, max_value):
    successes = 0
    trials = 0
    for cards in get_all_card_possibilities(n_cards, min_value, max_value):
        # results = get_possible_results(cards)
        # if target in results:
        #     successes += 1
        if target_is_possible(target, cards):
            successes += 1
        trials += 1
    return successes / trials


if __name__ == "__main__":
    n_cards = 4
    min_value = 1
    max_value = 13
    # cards = [random.randrange(min_value, max_value+1) for i in range(n_cards)]
    # results = get_possible_results(cards)

    for target in range(1, 201):
        target_occurrence = get_target_occurrence(target, n_cards, min_value, max_value)
        print(f"target {target} is possible {target_occurrence:.4f} of the time")
