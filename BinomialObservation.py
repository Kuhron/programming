import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import math


class BinomialObservation:
    def __init__(self, successes=0, trials=0):
        self.successes = successes
        self.trials = trials

    def __repr__(self):
        ns = self.successes
        nt = self.trials
        p = self.get_probability_estimator()
        w1, w2 = self.get_wilson_ci(0.95)
        return f"<obs {ns}/{nt} ({w1:.4f}, {p:.4f}, {w2:.4f}) 95% CI>"

    def get_probability_estimator(self):
        return self.successes / self.trials

    def get_normal_approximation_ci(self, confidence_level):
        z = BinomialObservation.get_z(confidence_level)
        s = self.successes
        n = self.trials
        p_hat = s/n
        error = z * math.sqrt((p_hat * (1-p_hat))/n)
        return (p_hat - error, p_hat + error)

    @staticmethod
    def get_z(confidence_level):
        # z is the 1 - alpha/2 quantile of N(0,1)
        alpha = 1 - confidence_level
        quantile = 1 - alpha/2
        z = scipy.stats.norm.ppf(quantile)
        return z

    def get_wilson_ci(self, confidence_level):
        z = BinomialObservation.get_z(confidence_level)
        s = self.successes
        n = self.trials
        p_hat = s/n
        first_term = (1/(1+z**2/n)) * (p_hat + z**2/(2*n))
        second_term = (z/(1+z**2/n)) * math.sqrt((p_hat * (1-p_hat) / n) + (z**2/(4*n**2)))
        return (first_term - second_term, first_term + second_term)

    def get_wilson_width(self, confidence_level):
        w1, w2 = self.get_wilson_ci(confidence_level)
        return w2 - w1

    def get_centered_wilson_estimator(self, confidence_level):
        # 1-p of the way from lower to upper bound, so it's more toward 50% to reflect uncertainty
        s = self.successes
        n = self.trials
        p_hat = s/n
        lower, upper = self.get_wilson_ci(confidence_level)
        return lower + (1-p_hat) * (upper-lower)

    def choose_random_possible_probability(self):
        # the real way to do this fairly would be invert the Wilson CI as function of confidence, normalize that integral to 1,
        # and choose a binomial p from that (i.e., weighted choice of probability by how much it shows up in confidence intervals)
        # but I'm not gonna do that math right now
        # doing it by choosing random confidence uniformly and then uniform p within that, overweights the p near the estimator, but I'm fine with that for now
        confidence = random.random()
        wci = self.get_wilson_ci(confidence)
        return random.uniform(*wci)


def wilson_function(successes, trials, confidence, alpha=0):
    # alpha means how far you go between the lower bound and the upper bound of the Wilson CI
    binom = BinomialObservation(successes, trials)
    lb, ub = binom.get_wilson_ci(confidence)
    return lb + alpha*(ub-lb)


if __name__ == "__main__":
    # plot Wilson CI as continuous function
    successes = np.linspace(0, 5, 100)
    trials = np.linspace(0.01, 5, 100)
    Suc, Tri = np.meshgrid(successes, trials)
    valid_mask = (Suc <= Tri)
    confidence = 0.1
    alpha = 0
    Z = np.zeros(valid_mask.shape)
    Z[valid_mask] = np.vectorize(lambda suc, tri: wilson_function(suc, tri, confidence, alpha))(Suc[valid_mask], Tri[valid_mask])
    Z[~valid_mask] = np.nan
    plt.imshow(Z)
    plt.colorbar()
    plt.show()
