# experiment of how you would order the rationals in [0,1] based on their Wilson confidence interval values
# but note that 1/2 and 2/4 give different CI values because of the number of trials being different, so only do coprime pairs I guess

import numpy as np
import matplotlib.pyplot as plt
import math

from BinomialObservation import BinomialObservation


def get_coprime_tuples(max_denom):
    res = []
    for denom in range(1, max_denom+1):
        if denom == 1:
            res.append((0,1))
            res.append((1,1))
        else:
            for numer in range(1, denom):  # don't want value equaling either 0 or 1 with denom other than 1
                if math.gcd(numer, denom) == 1:
                    res.append((numer, denom))
    return res


if __name__ == "__main__":
    max_denom = 50
    confidence_level = 0.95
    lower_bound = False

    tups = get_coprime_tuples(max_denom)
    ps = []
    ys = []
    for tup in tups:
        ns, nt = tup
        obs = BinomialObservation(successes=ns, trials=nt)
        p = obs.get_probability_estimator()
        y_lower, y_upper = obs.get_wilson_ci(confidence_level)
        y = y_lower if lower_bound else y_upper
        ps.append(p)
        ys.append(y)

    plt.scatter(ps, ys)
    plt.show()

    new_ordering = [p for p,y in sorted(zip(ps, ys), key=lambda pair: pair[1])]
    print("new ordering of rationals in [0,1]:")
    for x in new_ordering:
        print(x)
