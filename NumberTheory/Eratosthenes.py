# trying to generate primes relatively efficiently
# idea: Eratosthenes up to some number, and keep results in a generator
# if user asks for beyond that, then do it again up to some higher number using the primes you already know

import math


STARTING_PRIMES = [2,3,5]


def eratosthenes_in_interval(a, b, known_primes):
    # interval a to b does not include b
    # when we get to where the next known prime is greater than the lowest remaining candidate, any such candidates are new primes and we need to use them to cross off as well
    cands = list(range(a, b))
    is_crossed_off = {x: False for x in cands}
    for p in known_primes:
        is_crossed_off = cross_off(p, a, b, is_crossed_off)

    # now we've finished crossing based on these known primes
    # next, go through the remaining candidates, lowest one is a prime, and cross off based on it, repeat for next lowest remaining after that, etc.
    new_primes = []
    for c in cands:
        if is_crossed_off[c]:
            continue
        else:
            new_primes.append(c)
            is_crossed_off = cross_off(c, a, b, is_crossed_off)

    # sort the two lists before merging them, hoping that they're already sorted so this won't take long
    known_primes = sorted(known_primes)
    new_primes = sorted(new_primes)
    known_primes_in_interval = [p for p in known_primes if a <= p < b]
    assert all(a <= p < b for p in new_primes)
    primes_in_interval = merge_in_order(known_primes_in_interval, new_primes, allow_duplicates=False)
    for p in primes_in_interval:
        yield p


def merge_in_order(l1, l2, allow_duplicates=True):
    # assumes each list is in order and we'll just pick off the min of their leftmost elements
    res = []
    while True:
        x1 = l1[0] if len(l1) > 0 else None
        x2 = l2[0] if len(l2) > 0 else None
        if x1 is None and x2 is None:
            break

        if not allow_duplicates:
            while x1 == x2:
                # after check for None, so we know they're both numbers
                # get rid of x2
                l2 = l2[1:]
                x2 = l2[0]  # the new leftmost
        if x1 is None and x2 is not None:
            res.append(x2)
            l2 = l2[1:]
        elif x1 is not None and x2 is None:
            res.append(x1)
            l1 = l1[1:]
        elif x1 <= x2:
            res.append(x1)
            l1 = l1[1:]
        else:
            res.append(x2)
            l2 = l2[1:]
    return res


def cross_off(p, a, b, is_crossed_off):
    least_m = math.ceil(a / p)
    most_m = math.floor((b-1)/p)
    for m in range(least_m, most_m + 1):
        x = p * m
        is_crossed_off[x] = True
        # so we're mutating the input, yeah, but I don't want to make another copy of the dict every time
    return is_crossed_off


def get_all_primes():
    # progressively expand sieve of Eratosthenes
    known_primes = set()
    start = 2
    end = 1000
    while True:
        g = eratosthenes_in_interval(start, end, known_primes)
        ps = list(g)
        for p in ps:
            yield p
        known_primes |= set(ps)
        start = end + 1
        end *= 2


g = get_all_primes()
for i,p in enumerate(g):
    if i >= 1000:
        break
    print(p)


