# verifying that my favorite method for mental number generation is sufficiently unbiased


import random
import string
from collections import defaultdict

import matplotlib.pyplot as plt


with open("words.txt") as f:
    lines = f.readlines()

print(len(lines))

letter_value = lambda x: next(((i + 1) for i, item in enumerate(string.ascii_lowercase) if item == x), 0)
word_value = lambda w: sum(letter_value(x) for x in w)

n_choices = random.randint(2, 30)
print("n_choices:", n_choices)
n_trials = 100000

counter = defaultdict(int)

for i in range(n_trials):
    w = random.choice(lines)
    val = word_value(w) % n_choices  
    # actual mental method is to mod each letter and then add to the sum, and mod the sum if it grows too much in the middle, but this is equivalent
    counter[val] += 1

for k, v in counter.items():
    counter[k] /= n_trials


print(counter)

xs, ys = zip(*(counter.items()))
plt.plot(xs, ys)
plt.ylim(0, 1.1 * max(ys))  # don't be misleading about variance
plt.show()
