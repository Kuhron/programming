import math
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNC


def get_random_point(dimensions):
	return np.random.normal(0, 1, dimensions)

def get_classification(point):
	x, y = point
	return 0 if x**2 + y**2 + random.uniform(-0.3, 0.3) < 0.8 else 1

def get_sample(dimensions):
	return [get_random_point(dimensions) for i in range(1000)]


dimensions = 2
sample = get_sample(dimensions)

xs = [i[0] for i in sample]
ys = [i[1] for i in sample]
classes = [get_classification(i) for i in sample]
colors = ["r" if i == 0 else "b" for i in classes]
plt.scatter(xs, ys, color=colors)
plt.show()

clf1 = KNC(10, weights="distance")
clf2 = KNC(10, weights="uniform")

clfs = [clf1, clf2]
for clf in clfs:
	clf.fit(sample, classes)

new_sample = get_sample(dimensions)
new_classes = [get_classification(i) for i in new_sample]
for clf in clfs:
	print(clf.score(new_sample, new_classes))

a1 = []
a2 = []
for k in range(1, 50):
	clf1 = KNC(k, weights="distance")
	clf2 = KNC(k, weights="uniform")
	clf1.fit(sample, classes)
	clf2.fit(sample, classes)
	a1.append(clf1.score(new_sample, new_classes))
	a2.append(clf2.score(new_sample, new_classes))

plt.plot(a1, label="distance-weighted")
plt.plot(a2, label="uniform-weighted")
plt.legend()
plt.show()