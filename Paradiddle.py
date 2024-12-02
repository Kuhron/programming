import numpy as np
import matplotlib.pyplot as plt
import math


def step_digits(digs, base):
    if len(digs) == 0:
        # need this for recursive step to be able to add a new leading 1
        return [1]
    elif digs[-1] == base-1:
        return step_digits(digs[:-1], base=base) + [0]
    else:
        return digs[:-1] + [digs[-1] + 1]


def get_current_char(digs, base_pattern, chars):
    polarity = get_pattern_polarity(digs, base_pattern)
    return chars[polarity]


def digits_have_same_pattern_polarity(digs1, digs2, base_pattern):
    # know the changes are occurring at the right edge because digs is big-endian
    # find the prefix in the DIGITS (not the bits) that hasn't changed and ignore it
    # remember that different digits can lead to the same bits because base_pattern has various 0s and 1s
    n = max(len(digs1), len(digs2))
    digs1 = [0] * (n - len(digs1)) + digs1
    digs2 = [0] * (n - len(digs2)) + digs2
    first_disagreement_index = None
    for i in range(n):
        d1 = digs1[i]
        d2 = digs2[i]
        if d1 != d2:
            first_disagreement_index = i
            break
    if first_disagreement_index is None:
        # the digs are the same
        return True
    else:
        # check the polarity of the rest of them
        p1 = get_pattern_polarity(digs1[first_disagreement_index:], base_pattern)
        p2 = get_pattern_polarity(digs2[first_disagreement_index:], base_pattern)
        return p1 == p2


def get_pattern_polarity(digs, base_pattern):
    # each 1 at a fractal level that we're in says to flip polarity
    # (also why we start the base pattern at 0 not 1, so all points in time have digit sequence with infinite leading 0s rather than leading 1s)
    return sum(base_pattern[i] for i in digs) % 2


def get_chars(base_pattern, chars, buffer_length=1):
    g = get_bits(base_pattern)
    while True:
        s = ""
        for i in range(buffer_length):
            b = next(g)
            s += chars[b]
        yield s


def get_bits(base_pattern):
    # infinite generator
    assert base_pattern[0] == 0
    assert all(x in [0, 1] for x in base_pattern)
    digit_base = len(base_pattern)  # each level of the fractal keeps track of what part of the base pattern we're currently in at that level
    digs = [0]
    old_digs = None
    current_polarity = 0
    while True:
        # step digs no matter what, then determine as quickly as possible whether the polarity has changed
        polarity_changed = (old_digs is None) or (not digits_have_same_pattern_polarity(digs, old_digs, base_pattern))

        if polarity_changed:
            current_polarity = 1 - current_polarity
        yield current_polarity

        old_digs = digs
        digs = step_digits(digs, base=digit_base)


def print_fractal(base_pattern, chars, buffer_length):
    g = get_chars(base_pattern, chars, buffer_length)
    while True:
        input(next(g))


def plot_fractal(base_pattern, n_points):
    g = get_bits(base_pattern)
    bits = np.array([next(g) for i in range(n_points)])
    bits = 1+(2*(bits-1))
    weights = get_trajectory_contribution_array(n_points, len(base_pattern))
    trajectory = np.cumsum(bits * weights)

    # TODO I don't like this trajectory for [0, 1, 0, 0] very much, I see what it's doing but I wanted the 0 segments to be at the same height as each other and the 1 at another height, and then each of those segments is a copy of the fractal (inverted in a 1 segment), figure out how to do this

    plt.plot(trajectory)
    plt.show()


def get_trajectory_contribution(index, base):
    # give more weight to indices at powers of the base pattern length
    # so I can get a nice cumulative sum fractal rather than everything being only 0 or 1, which doesn't look good even though it has the same fractal structure in 1D
    # i=0 is the first element, should get weight of 1
    # the first element at a power of `base` gets weight of 2, and so on (exponential)
    return 2**math.floor(math.log(index+1, base))


def get_trajectory_contribution_array(n_points, base):
    # fill them in like a sieve rather than calculating logs at each i
    a = np.full((n_points,), 1)
    p = 1
    bp = base**p
    factor_base = 2
    factor = factor_base**p
    while bp < n_points:
        # fill in the slots
        m = 1
        while m * bp < n_points:
            a[m*bp] = factor
            # I guess I'm doing extra work overwriting ones that actually have a higher power but whatever
            m += 1

        p += 1
        bp = base**p
        factor = factor_base**p
    return a


if __name__ == "__main__":
    chars = "+-"
    # base_pattern = [0, 1, 1, 0]
    base_pattern = [0, 1, 0, 0]

    buffer_length = 64  # also try not a power of 2 to see what it looks like

    # print_fractal(base_pattern, chars, buffer_length)
    plot_fractal(base_pattern, n_points=2**16)

