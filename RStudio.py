# I wonder if I can write a Python script in RStudio

# this is my typical black-swan generating algorithm

import random
import matplotlib.pyplot as plt

def r(precision):
    v = 1
    while random.random() < 1-(1.0/precision): # e.g. precision = 100, this = 0.99
        v *= 1/(1-1.0/precision)
    return v

a = [r(100) for i in range(10000)]
plt.plot(a)
plt.show()