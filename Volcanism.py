# simulate volcanism or other events that can be thought of similar to pressure building and then being released with certain probability

import random

import matplotlib.pyplot as plt


def eruption_pdf(pressure, threshold):
    if pressure < threshold:
        # threshold for any eruption
        return 0
    return random.choice(range(pressure))

systematic_increase = 100
threshold = 50000
initial_pressure = threshold-systematic_increase

p = initial_pressure
p_list = [initial_pressure]
e_list = [0]
for i in range(10000):
    p += systematic_increase
    eruption = eruption_pdf(p,threshold)
    p -= eruption
    p_list.append(p)
    e_list.append(eruption)

plt.plot(e_list)
plt.show()