import math
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Vector:
	def __init__(self, array):
		self.array = array
		self.magnitude = math.sqrt(sum([i**2 for i in array]))
		self.unit_vector = [i * 1.0/self.magnitude for i in array]
		self.dimensions = len(array)

	def __repr__(self):
		return repr(self.array)

	def __getitem__(self, key):
		return self.array[key]


def get_random_vector(dimensions):
	return Vector(np.random.normal(0, 1, dimensions))

def get_closest_point_on_line(point, line):
	# treating line as infinite extension of a normalized vector, and we can get the closest point in any plane
	pass

def dilate_along_line(point, coefficients):
	pass


def get_explained_variance(outlier):
	dimensions = 5
	sample = np.array([get_random_vector(dimensions).array for i in range(10**4)] + [np.array([outlier for i in range(dimensions)])])
	x, y = random.sample(range(dimensions), 2)

	xs = [i[x] for i in sample]
	ys = [i[y] for i in sample]

	# plt.scatter(xs, ys)
	# plt.show()

	pca = PCA(dimensions)
	pca.fit(sample)
	# print("components:\n{0}".format(pca.components_))
	# print("explained variance ratio:\n{0}".format(pca.explained_variance_ratio_))
	return pca.explained_variance_ratio_[0]


a = []
for outlier in range(0, 500, 5):
	print(outlier)
	a.append(get_explained_variance(outlier))

plt.plot(a)
plt.show()