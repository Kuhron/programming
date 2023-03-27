# trying to model processes where something flips back and forth between 0 and 1 at random
# geomagnetic reversal, a light flickering toward the end of its life, etc.

# should have the property that both poles are stable but will allow fluctuation that can magnify into a flip
# allow it to take on values in the interval [0, 1] and graph these over time
# ideally it will be symmetric, 0 <-> 1

# try getting it to happen from some underlying mechanism, like the voltage and current changing, rather than modeling the time series directly
# make it as deterministic as possible rather than adding any random noise, want noise to be endogenous from initial conditions and equations


import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
ln = math.log
exp = math.exp


def print_charges(charges):
    s = ""
    for i in range(10, -1, -1):
        s += str(i).rjust(2) + " " + " ".join(["|" if c >= i/10 else " " for c in charges]) + "\n"
    s += " "*3 + " ".join("0123456789AB") + "\n"
    print(s)


def evolve_charges(charges, temperatures, conductances):
    effective_conductances = [get_effective_conductance(c, t) for c, t in zip(conductances, temperatures)]
    n = len(conductances)
    # each charge tries to move to the right
    # the fraction of the charge that can move is the effective conductance of this cell
    # the fraction of that that is accepted is the effective conductance of the next cell
    # so take product of those two conductances
    charge_fractions_to_move = [effective_conductances[i] * effective_conductances[(i+1) % n] for i in range(n)]
    charge_to_move = [charges[i] * charge_fractions_to_move[i] for i in range(n)]
    for i in range(n):
        charges[i] -= charge_to_move[i]
        charges[(i+1) % n] += charge_to_move[i]
    assert all(0 <= c <= 1 for c in charges)

    # then the temperature will change somehow
    # the more charge moved through the cell, the hotter it will get
    # if no charge is moving through it, it will fall in temperature back toward some baseline
    t0 = 0.5
    fall_fraction = 0.1
    rise_responsiveness = 0.5
    charge_moved = [charge_to_move[i] + charge_to_move[(i-1) % n] for i in range(n)]
    max_rises = [1-temperatures[i] for i in range(n)]
    rises = [max_rises[i] * charge_moved[i] * rise_responsiveness for i in range(n)]
    max_falls = [temperatures[i] - t0 for i in range(n)]
    falls = [fall_fraction * max_falls[i] for i in range(n)]
    for i in range(n):
        temperatures[i] += rises[i] - falls[i]

    return charges, temperatures


def get_effective_conductance(c, t):
    # c is baseline, higher temperature makes it conduct more
    # both values should always be in interval [0, 1]
    assert 0 <= c <= 1
    assert 0 <= t <= 1
    # at temperature 0.5, return baseline conductance
    # sigmoid from [0,1] to itself
    # https://www.desmos.com/calculator/mzeb9wmrgp
    a = 2  # temperature responsiveness
    x = t

    # b = ln(1/c - 1)
    # g = a*ln(1/x - 1) + b
    # y = 1/(1 + exp(g))

    # simplified expression
    y = 1/(1 + (1/x - 1)**a * (1/c-1))

    assert 0 <= y <= 1
    return y



if __name__ == "__main__":
    conductances = [0.4, 0.71, 0.12, 0.35, 0.84, 0.9, 0.51, 0.07, 0.24, 0.26, 0.77, 0.56]
    n_cells = len(conductances)
    temperatures = [0.5] * n_cells

    # the hotter a piece of wire is, the more it can conduct
    # but put some lag times in the changes, like heating up takes time
    # pass charge from left to right

    charges = [1/n_cells for i in range(n_cells)]

    current_history = [[0 for x in charges]]
    temperature_history = [[x for x in temperatures]]

    n_steps = 250000
    for i in range(n_steps):
        if i % 100 == 0:
            print(f"{i}/{n_steps}")
        last_charges = [x for x in charges]
        charges, temperatures = evolve_charges(charges, temperatures, conductances)
        current = [charges[i] - last_charges[i] for i in range(n_cells)]

        # os.system("clear")
        # print_charges(charges)
        # time.sleep(0.01)

        current_history.append([x for x in current])
        temperature_history.append([x for x in temperatures])

    left = bottom = 0
    right = n_cells
    top = n_steps
    extent = (left, right, bottom, top)
    aspect = n_cells / n_steps

    plt.subplot(1,2,1)
    plt.imshow(current_history, aspect=aspect)
    plt.title("current")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(temperature_history, aspect=aspect)
    plt.title("temperature")
    plt.colorbar()

    plt.show()

