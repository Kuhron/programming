import math
import random

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, svm


def get_accuracy(data, target, n_training, C):
	indices = [i for i in range(len(data))]
	random.shuffle(indices)
	data = [data[i] for i in indices]
	target = [target[i] for i in indices]
	training_data = np.array(data[:n_training])
	training_target = np.array(target[:n_training])
	testing_data = np.array(data[n_training:])
	testing_target = np.array(target[n_training:])

	classifier = svm.SVC(kernel="linear", C=C)
	classifier.fit(training_data, training_target)

	correct_predictions = 0
	for i in range(len(testing_data)):
		prediction = classifier.predict(testing_data[i].reshape(1, -1))
		if prediction == testing_target[i]:
			correct_predictions += 1
	return correct_predictions * 1.0/len(testing_data)

def test():
	n = 100
	x = [random.random() for i in range(n)] # [1, 5, 1.5, 8, 1, 9]
	y = [random.random() for i in range(n)] # [2, 8, 1.8, 8, 0.6, 11]

	data = np.array([i for i in zip(x, y)])
	print(data.shape)
	target = np.array([(0 if y[i] < x[i] else 1) for i in range(n)]).reshape(-1, 1) # [0, 1, 0, 1, 0, 1]
	print(target.shape)

	classifier = svm.SVC(kernel="linear", C=1.0)
	classifier.fit(data, target)

	a1 = np.array([0.58, 0.76])#.reshape(-1, 1)
	a2 = np.array([1000, 1000])#.reshape(-1, 1)
	print(classifier.predict(a1))
	print(classifier.predict(a2))


digits = datasets.load_digits()
data = digits.data
target = digits.target

accuracies = []
log_inaccuracies = []
length = 20
for n in np.arange(10, len(data)-1, len(data)*1.0/length):
	n = int(n)
	accuracies.append([])
	log_inaccuracies.append([])
	for p in np.arange(-7, 0, 7*1.0/length):
		C = 10**p
		accuracy = get_accuracy(data, target, n, C)
		print("C = {C:.8f}, n = {n:4d} gave accuracy {accuracy}".format(C=C, n=n, accuracy=accuracy))
		accuracies[-1].append(accuracy)
		log_inaccuracies[-1].append(math.log(1-accuracy) if accuracy != 1 else -10)

im = plt.imshow(accuracies)
plt.title("accuracy")
plt.colorbar(im, orientation="horizontal")
plt.show()
plt.close()

im = plt.imshow(log_inaccuracies)
plt.title("natural log of (1-accuracy)")
plt.colorbar(im, orientation="horizontal")
plt.show()
plt.close()

























