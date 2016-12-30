import math
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


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

clf = SVC(kernel="rbf")
clf.fit(sample, classes)

new_sample = get_sample(dimensions)
new_classes = [get_classification(i) for i in new_sample]
print(clf.score(new_sample, new_classes))

