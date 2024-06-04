# look at how the possible prime powers "span" the integers
# and what would happen if you used another set of numbers, instead of the primes, as the factors

import math
import random
import itertools
import sympy as sp
from sympy.ntheory import factorint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def traverse_powers(n_primes):
    # there are probably a lot of ways I could order these
    # I want this function to give, in some order, the arrays like:
    #   [1], [0, 1], [1, 1], [0, 0, 1], [2], [2, 1], [0, 2], [0, 1, 2], ...
    # which will be the exponents on the primes, so [2, 1] means 2^2 * 3 = 12

    return traverse_powers_length_equals_base(n_primes)
    # put alternate options for traversal order here (calling other functions)


def traverse_powers_length_equals_base(n_primes):
    for l in range(n_primes + 1):
        # in this round we are allowed to use digits 0..l

        # hardcode simple base cases
        if l == 0:
            yield []
            continue
        elif l == 1:
            yield [1]
            continue

        # for n_primes >= 2 we can use general procedure (probably reasonably easy with either recursion or iteration, from looking at the patterns in the arrays that I wrote out by hand)

        # first fill in anything that was "missed" last round because we had access only to digits 0..l-1, so keep length strictly less than l (not counting leading zeros) and create everything that has an l somewhere
        # binary array of where the new digit is located (can be in more than one place but must be in at least one)
        # e.g. for doing l=3 but putting 3 into len-2 arrays (still little-endian like these arrays all are): 3x, x3, 33
        # within each of these, go over the possibilities for x in 0..l-1 in order

        digit_position_arrays = [x[::-1] for x in itertools.product(*([[0,l]]*(l-1)))][1:]
        # use l, rather than 1, to represent where l is located (could do either but I like this way)
        # the first item in the product is array of all zeros, which we don't want because there is no l in it

        digits_to_fill_in_slots = list(range(l))  # excluding l
        for a in digit_position_arrays:
            for b in fill_out_digit_position_array(a, digits_to_fill_in_slots):
                yield b

        # now start with 0, 0, 0, ..., 1 (of length l) and fill in everything with digits 0..l
        digits_to_fill_in_slots = list(range(l+1))
        position_array = tuple(0 for i in range(l))  # wherever there is a zero in the position array, all values in the filling digits can be placed
        for b in fill_out_digit_position_array(position_array, digits_to_fill_in_slots):
            if b[-1] != 0:  # easier to just check this here
                yield b


def fill_out_digit_position_array(position_array, digits_to_fill_in_slots, fillable_slot_value=0):
    # wherever the array has the fillable slot value, put each of the values in digits_to_fill_in_slots
    assert type(digits_to_fill_in_slots) is list
    is_slot = [x == fillable_slot_value for x in position_array]
    slot_indices = [i for i,x in enumerate(is_slot) if x is True]
    n_slots = sum(is_slot)

    if n_slots == 0:
        # only one way to fill zero slots
        yield list(position_array)
    else:
        a = [x for x in position_array]
        for combo in itertools.product(*([digits_to_fill_in_slots]*n_slots)):
            for i, c in enumerate(combo[::-1]):
                slot_index = slot_indices[i]
                a[slot_index] = c
            yield a


def get_product(primes, powers):
    n = 1
    for p, x in zip(primes, powers):
        n *= (p**x)
    return n


def plot_number_counts(number_counts, connect_dots=False, y_scale_log=True):
    # connect_dots is just to make it look cool
    ns = []
    counts = []
    for n,count in sorted(number_counts.items()):
        ns.append(n)
        counts.append(count)
    plt.scatter(ns, counts)
    plt.xscale("log")
    if y_scale_log:
        plt.yscale("log")

    if connect_dots:
        # connect each dot to the nearest dot on the level above and below it
        count_to_ns = {}
        for n,count in zip(ns, counts):
            if count not in count_to_ns:
                count_to_ns[count] = []
            count_to_ns[count].append(math.log(n))
            # ns should already be sorted on each level

        levels = sorted(set(counts))  # don't overwrite variable name, although I guess I'm done with `counts` now
        if len(levels) > 1:
            # can probably come up with some nice algorithms for finding the connection lines, but start simple and inefficient (but not stupidly so)
            link_lines = []
            for l_i in range(len(levels)):
                ref_level = levels[l_i]
                if l_i == 0:
                    link_levels = [levels[1]]
                elif l_i == len(levels) - 1:
                    link_levels = [levels[-2]]
                else:
                    link_levels = [levels[l_i-1], levels[l_i+1]]

                for link_level in link_levels:
                    # get the lines connecting ref_level to link_level
                    xs_on_ref_level = count_to_ns[ref_level]
                    xs_on_link_level = count_to_ns[link_level]
                    levels_by_x = {}
                    for x in xs_on_ref_level:
                        levels_by_x[x] = ["ref"]
                    for x in xs_on_link_level:
                        if x in levels_by_x:
                            raise Exception("shouldn't happen; each x should be on only one level because each number occurs some number of times, not two numbers of times")
                            # levels_by_x[x].append("link")
                        else:
                            levels_by_x[x] = ["link"]
                    xs_in_order = sorted(levels_by_x.keys())
                    # for each x on ref level, find nearest x on link level in both directions
                    # go over all the xs in order so we know what index we're at and can easily get neighboring values on the other level
                    for x_i, x in enumerate(xs_in_order):
                        if "ref" not in levels_by_x[x]:
                            continue

                        # link to closest thing on the left, either on ref_level or link_level
                        ref_point = [math.exp(x), ref_level]
                        left_x = xs_in_order[x_i - 1] if x_i > 0 else None
                        right_x = xs_in_order[x_i + 1] if x_i < len(xs_in_order) - 1 else None
                        if left_x is not None:
                            if levels_by_x[left_x] == ["ref"]:
                                left_level = ref_level
                            elif levels_by_x[left_x] == ["link"]:
                                left_level = link_level
                            else:
                                raise Exception(levels_by_x[left_x])
                            left_link_point = [math.exp(left_x), left_level]
                            link_lines.append([ref_point, left_link_point])
                        if right_x is not None:
                            if levels_by_x[right_x] == ["ref"]:
                                right_level = ref_level
                            elif levels_by_x[right_x] == ["link"]:
                                right_level = link_level
                            else:
                                raise Exception(levels_by_x[right_x])
                            right_link_point = [math.exp(right_x), right_level]
                            link_lines.append([ref_point, right_link_point])

                        """
                        # go left
                        left_link = None
                        x_j = x_i - 1
                        while x_j >= 0:
                            x2 = xs_in_order[x_j]
                            assert x2 < x
                            if "link" in levels_by_x[x2]:
                                left_link = x2
                                break
                            x_j -= 1
                        # if it finds nothing on link level to left of this x, then left_link remains None

                        # go right
                        right_link = None
                        x_j = x_i + 1
                        while x_j < len(xs_in_order):
                            x2 = xs_in_order[x_j]
                            assert x2 > x
                            if "link" in levels_by_x[x2]:
                                right_link = x2
                                break
                            x_j += 1
                        # if it finds nothing on link level to right of this x, then right_link remains None

                        if left_link is not None:
                            # make a line segment
                            start_point = [math.exp(x), ref_level]
                            end_point = [math.exp(left_link), link_level]
                            link_lines.append([start_point, end_point])
                        if right_link is not None:
                            # make a line segment
                            start_point = [math.exp(x), ref_level]
                            end_point = [math.exp(right_link), link_level]
                            link_lines.append([start_point, end_point])
                        """

            collection = LineCollection(link_lines)
            plt.gca().add_collection(collection)

    plt.show()


if __name__ == "__main__":
    # ps = list(sp.primerange(0, 12))  # only primes -> each number appears at most once (FTO Arithmetic)
    # ps = [2,3,4,5,6]
    # ps = [2,4,8,16,32]  # arch
    # ps = [2,3,6,8,12,15]
    # ps = [2,3,4,8,16]  # arch slid horizontally because I added something coprime with 2
    # ps = [2,4,8,16,17]  # arch slid/copied horizontally with wider steps
    # ps = [2,3,4,5,6]
    # ps = [3,5,9,25,27]  # cool tower with sub-arches
    ps = [4,5,16,25,64]

    print(f"{ps = }")

    power_arrays = traverse_powers(len(ps))
    arrays_seen = set()
    number_counts = {}
    for i, a in enumerate(power_arrays):
        a_tup = tuple(a)
        if a_tup in arrays_seen:
            raise Exception(f"traversal gave duplicate power array: {a}")
        arrays_seen.add(a_tup)

        n = get_product(ps, a)
        if n not in number_counts:
            number_counts[n] = 0
        number_counts[n] += 1

        # print(i, a, n)

    # for k,v in sorted(number_counts.items()):
    #     print(k, v)

    # plot some stats about what numbers were created how many times
    plot_number_counts(number_counts, y_scale_log=False, connect_dots=True)

    # observations / questions:
    # - the plot of log(n) vs log(count) has bilateral symmetry, why?
    # - setting ps to the first n powers of 2 makes the graph an arch
    # - sometimes not everything in ps is coprime to everything else, but you still get every number being generated at most once, how? maybe it's that none of them share *all* the same prime factors
    # 

