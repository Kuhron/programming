# what happens when you iterate Bayes' Theorem on every possible initial prior?
# take interval (0, 1) excluding 0 and 1 themselves (because they won't change regardless of evidence)
# using same series of evidences, update probabilities and see whether the interval converges to the correct value over time, how long it takes to get to a certain precision, etc.

import random
import numpy as np
import matplotlib.pyplot as plt


def get_new_probability(pa, pb, pb_a):
    return pb_a * pa / pb


if __name__ == "__main__":
    # need two axes, two binary variables
    
    p_heads = 0.75
    prior = 0.5
    for i in range(100):
        heads = random.random() < p_heads
        pb_a = 
        prior = get_new_probability(pa=prior, pb=(1-prior), pb_a=pb_a)
