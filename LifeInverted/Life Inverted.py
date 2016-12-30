import os
import random
import time

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import numpy as np
from scipy.signal.signaltools import correlate2d


def step(X, n_step=1):
	# pass a bool array
	# uses toroidal array
	# could be more efficient by not shifting the whole array (np.roll(X))?
	if n_step > 1:
		X = step(X, n_step-1)
	elif n_step < 1:
		raise Exception("need at least one step")
	nc = nbrs_count(X)
	result = (nc == 3) | (X & (nc == 2))
	return result

def nbrs_count(X):
	return sum(np.roll(np.roll(X, i, 0), j, 1)
               for i in (-1, 0, 1) for j in (-1, 0, 1)
               if (i != 0 or j != 0))

def augment(X):
	nc = nbrs_count(X)
	return X | (nc > 0)

def fizzle(X):
	Y = np.random.choice([False, True], size=X.shape)
	return X & Y

def get_points_in_neighborhoods(X, bool_value):
    nc = nbrs_count(X)
    if bool_value:
    	return (X & (nc == 8))
    else:
    	return (~X & (nc == 0))

def display(X):
	for row in X:
		for col in row:
			print("X" if col else "-", end="")
		print()
	print("\n")

def corr_score(array1, array2):
	return correlate2d(array1, array2, mode="same").max()

def simple_score(array1, array2):
	return np.mean(array1 == array2)

def mutate(array, flip=True, invert_probability=0.1):
	# if have actual image, make this smart to know only to add True if a cell next to it is True, etc. to narrow search space
	# to flip certain elements only, xor an array with an array where True is the positions to negate
	negation = np.random.choice([True, False], size=array.shape, p=[invert_probability, 1 - invert_probability])
	x, y = array.shape
	landlocked_True = get_points_in_neighborhoods(array, True)
	landlocked_False = get_points_in_neighborhoods(array, False)
	doubly_landlocked_True = get_points_in_neighborhoods(landlocked_True, True)
	doubly_landlocked_False = get_points_in_neighborhoods(landlocked_False, True)
	negation = negation & ~doubly_landlocked_False  # don't invert points out in the large False regions
	if flip and random.random() < 0.5:
	    return (~array) ^ negation
	return array ^ negation


blinker = [1, 1, 1]
box = [[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1]]

# X = np.zeros((7, 7))
# X[3, 2:5] = blinker
# X = X.astype(bool)
# print(step(X))

# X = np.random.choice([0, 1], size=(20, 50))
# X = X.astype(bool)
# for i in range(1000):
# 	display(X)
# 	os.system("cls")
# 	X = step(X)

# target = np.random.choice([True, False], size=(20, 50))
# target = np.array([np.array([i % 2 == 0]*50) for i in range(20)])
# target[10:15, 30:37] = np.array(box).astype(bool)

files_and_cutoffs = {"kanye2.png": 5, "kanye2_large.png": 15, "kanye3.png": 50}
file_to_use = "kanye2_large.png"
# im = Image.open("img_test.png")
# im = Image.open("kanye.png")
# im = Image.open("kanye2.png")
im = Image.open(file_to_use)
im_grey = im.convert('L') # convert the image to *greyscale*
im_array = np.array(im_grey)
raw_target = im_array < files_and_cutoffs[file_to_use]  # 255 is white, set the cutoff manually to get a good image (75 or lower is good for kanye2 due to the darkness; even 5 is good)
landlocked_True = get_points_in_neighborhoods(raw_target, True)
landlocked_False = get_points_in_neighborhoods(raw_target, False)
doubly_landlocked_True = get_points_in_neighborhoods(landlocked_True, True)
doubly_landlocked_False = get_points_in_neighborhoods(landlocked_False, True)
# target = augment(raw_target) & ~doubly_landlocked_True  # just get True points in border regions between True and False
# target = fizzle(augment(raw_target & ~landlocked_True))
target = raw_target & ~landlocked_True
# target = raw_target  # just try to get the original boolean array
# target = doubly_landlocked_False
# print(im_array[100][100:120])
# plt.imsave('target.png', target.astype(int), cmap=cm.gray)
cmap = colors.ListedColormap(['red', 'blue'])  # maps to (False, True)
plt.imsave('raw_target.png', raw_target.astype(int), cmap=cmap)
plt.imsave('target.png', target.astype(int), cmap=cmap)
plt.imsave('target_nbrs_count.png', nbrs_count(target).astype(int), cmap=cm.gray)

# print("target:")
# display(target)

best_candidate = None
best_candidate_step = None
best_score = None
good_match_found = False
for i in range(100):
	if i % 10 == 0:
		print("iteration", i, end="\r")
	# if i <= 100:
	if best_score is None:
		candidate = mutate(target, flip=False, invert_probability=0.1)
	elif best_score < 0.92: # for simple_score, reject everything that is nowhere close
		# try some mutations first, then build stepwise
		# candidate = np.random.choice([True, False], size=target.shape)
		candidate = mutate(target, flip=False, invert_probability=0.1)
	else:
		if not good_match_found:
			# print("\ngood match! at iteration", i)
			good_match_found = True
		# candidate = best_candidate # just stays at first iteration
		candidate = mutate(best_candidate, flip=False, invert_probability=0.01) # better to start from target itself for zero steps
		# candidate = mutate(target) # actually gets good results for similarity with zero steps
	result = step(candidate, n_step=1) # use this when running for real
	# result = candidate # just for test of similarity
	score = simple_score(target.astype(int), result.astype(int))
	# score = corr_score(target.astype(int), result.astype(int))
	if best_candidate is None or score > best_score:
		best_candidate = candidate
		best_candidate_step = result
		best_score = score
print()

# print(best_corr)
# display(best_candidate)
# display(step(best_candidate))
# im_candidate = Image.fromarray(best_candidate.astype(int))
# im_candidate.save("candidate.png")
# im_candidate_step = Image.fromarray(step(best_candidate).astype(int))
# im_candidate_step.save("candidate_step.png")

plt.imsave('candidate.png', best_candidate.astype(int), cmap=cmap)
current_image = best_candidate
for i in range(1, 31):
	print("saving image {0}".format(i), end="\r")
	current_image = step(current_image)
	plt.imsave('candidate_step_{0}.png'.format(i), current_image.astype(int), cmap=cmap)
print("\ndone")