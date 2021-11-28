import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


b = round(np.random.normal(0, 10), 4)
envelope = -1/4 * (b**2)
m = round(random.uniform(20*envelope, envelope), 4)
c = round(np.random.normal(0, 20), 4)
n = 100

fdir = "RecurrenceRelationImages/"
fname = f"RecRelBcm_b{b}_m{m}_c{c}_n{n}.png"
fp = os.path.join(fdir, fname)
if os.path.exists(fp):
    print(f"image exists and will be overwritten: {fp}")
    inp = input("continue anyway? (default yes, enter any string to interrupt)")
    if inp != "":
        print("cancelled")
        sys.exit()

f = lambda a: b + m/a
seq = [c]
for i in range(n - 1):
    seq.append(f(seq[-1]))

xs = list(range(n))
ys = seq

plt.plot(xs, ys)
plt.scatter(xs, ys)
title = f"{b} + {m}/a_n; a_0 = {c}; {n} terms"
plt.title(title)
fig = plt.gcf()  # try to store it so it's not deleted after show()
plt.show()

if input("save plot? (default yes, enter any string to interrupt)") == "":
    fig.savefig(fp)
    print("saved")
else:
    print("not saved")
