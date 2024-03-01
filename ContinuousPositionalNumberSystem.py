# idea: trying to find supergradientist way to represent numbers
# colors on a canvas seems like a nice idea since the positions are continuous and so are the values at each position, and the color wheel wraps around so you don't have to have cutoffs/jumps/reversals due to there being a limit to how far you can go in one direction on the value axis
# but here I'm just going to do the same basic principle as a base system and positional digits, and take the limit as the positions get closer and closer together

import random
import math
import numpy as np
import matplotlib.pyplot as plt


def get_digits_no_gradience(n, b, dp):
    # dp is distance between powers
    # here we require non-negative integer for each digit
    log = math.log(n, b)
    # go left and right from log=0, so get the lowest multiple of dp needed to represent this power
    highest_p = dp * math.ceil(log/dp)

    # hack to get float precision estimate so it's not going all the way to b**-300
    n_digs = len(str(n).replace(".",""))
    low_log = highest_p - n_digs + 1
    lowest_p = dp * math.floor(low_log/dp)

    print(f"{log = }, {highest_p = }, {lowest_p = }")

    power_to_digit = {}
    p = highest_p
    rem = n
    while rem > 0:
        denom = b**p
        if denom == 0:
            # float precision limit
            break
        a = math.floor(rem / denom)
        if a != 0:
            power_to_digit[p] = a
        rem -= a * denom
        p -= dp
        if p < lowest_p:
            break
    for p, d in sorted(power_to_digit.items(), reverse=True):
        print(p, d)
    return power_to_digit


def get_digits_with_gradience(n, b, dp):
    # "digit" values are non-negative reals
    raise NotImplementedError

    # then decrease dp so you're basically taking the limit dp -> 0, and plot what the curve looks like of these real-valued digits over a continuous domain of powers


if __name__ == "__main__":
    # n = np.random.pareto(0.5) ** np.random.pareto(0.5)
    n = math.floor(random.random() * 1000000) * math.floor(random.random() * 1000000)
    print(n)

    b = 10
    dp = 0.9

    digs_no_grad = get_digits_no_gradience(n, b, dp)
    kmax = max(digs_no_grad.keys())
    kmin = min(digs_no_grad.keys())
    xs = list(np.arange(kmin, kmax+dp, dp))

    ys = []
    # stupid float dict keys
    min_key_diff = min(abs(x-y) for x in digs_no_grad.keys() for y in digs_no_grad.keys() if x != y)
    print(f"{min_key_diff = }")
    precision = min_key_diff * 0.49
    for x in xs:
        candidates = [k for k in digs_no_grad.keys() if abs(k-x) <= precision]
        if len(candidates) > 1:
            raise Exception("too many candidates")
        if len(candidates) == 1:
            y, = candidates
        else:
            # no digit for this power
            y = 0
        ys.append(y)
    plt.scatter(xs, ys)
    plt.show()

    # print(get_digits_with_gradience(n, b, dp))
