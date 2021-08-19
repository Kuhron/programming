# making different kinds of functions to evaluate dna strings by

import random
import numpy as np
import matplotlib.pyplot as plt


def plus_minus_cumsum(dna):
    # +1 for 1, -1 for 0
    xs = np.array([1 if x == 1 else -1 for x in dna])
    cumsum = xs.cumsum()
    cumsum = cumsum - cumsum.mean()  # stupid np -= casting crap
    return cumsum


def linear_choice_series(dna, coefficients=None, modification_function=None):
    a,b,c,d = np.random.normal(0,1,4) if coefficients is None else coefficients
    if modification_function is None:
        modification_function = lambda x: x
    x = 0
    xs = [x]
    for bit in dna:
        if bit == 0:
            x = a*x+b
        elif bit == 1:
            x = c*x+d
        else:
            raise ValueError
        x = modification_function(x)
        xs.append(x)
    return np.array(xs)


def signed_log(x):
    return np.sign(x) * np.log(1+abs(x))


def same_different_direction_path(dna):
    # 1 means switch directions and go 1 in the new direction
    # 0 means go 1 more in the same direction as before
    # initial direction is up
    val = 0
    arr = [val]
    direction = 1
    for bit in dna:
        if bit == 1:
            direction *= -1
        val += direction
        arr.append(val)
    return np.array(arr)


if __name__ == "__main__":
    raise Exception("don't run this, just import its functions")
