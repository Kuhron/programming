class BinomialObservation:
    def __init__(self, successes=0, trials=0):
        self.successes = successes
        self.trials = trials

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

    def choose_random_possible_probability(self):
        # the real way to do this fairly would be invert the Wilson CI as function of confidence, normalize that integral to 1,
        # and choose a binomial p from that (i.e., weighted choice of probability by how much it shows up in confidence intervals)
        # but I'm not gonna do that math right now
        # doing it by choosing random confidence uniformly and then uniform p within that, overweights the p near the estimator, but I'm fine with that for now
        confidence = random.random()
        wci = self.get_wilson_ci(confidence)
        return random.uniform(*wci)


