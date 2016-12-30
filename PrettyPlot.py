import math
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# plot curves by having black line where lhs==rhs and continuous coloring elsewhere

def f1(x, y, *args):
	r = args[0]

	lhs = x
	rhs = r * np.arccos(1-y/r) - np.sqrt(y*(2*r-y))

	print("f1; r = {r}".format(r=r).replace("; ", "\n   ")+"\n")
	return lhs - rhs

def f2(x, y, *args):
	# elliptic curve
	a = args[0]
	b = args[1]

	lhs = y**2
	rhs = x**3 + a*x + b

	print("f2; a = {a}; b = {b}".format(a=a, b=b).replace("; ", "\n   ")+"\n")
	return lhs - rhs

def f3(x, y, *args):
	a = args[0]
	b = args[1]

	lhs = y**2
	rhs = x**5 + a*abs(x)**0.5 + b

	print("f3; a = {a}; b = {b}".format(a=a, b=b).replace("; ", "\n   ")+"\n")
	return lhs - rhs

def f4(x, y, *args):
	n = len(args[0])
	a = args[0][:n]
	b = args[1][:n]
	c = args[2][:n]
	d = args[3][:n]
	e = args[4][:n]

	lhs = 0
	rhs = 0

	for i in range(len(a)):
		# the result is the sum of this many sine waves in 2d
		rhs += a[i]*np.sin(b[i]*x) + c[i]*np.sin(d[i]*y) + e[i]

	print("f4; a = {a}; b = {b}; c = {c}; d = {d}; e = {e}".format(a=a, b=b, c=c, d=d, e=e).replace("; ", "\n   ")+"\n")
	return lhs - rhs

def f5(x, y, *args):
	n = len(args[0])
	a = args[0][:n]
	b = args[1][:n]
	c = args[2][:n]
	d = args[3][:n]
	e = args[4][:n]

	lhs = 0
	rhs = 1

	for i in range(len(a)):
		# the result is the product of this many sine waves in 2d
		rhs *= a[i]*np.sin(b[i]*x) + c[i]*np.sin(d[i]*y) + e[i]

	print("f5; a = {a}; b = {b}; c = {c}; d = {d}; e = {e}".format(a=a, b=b, c=c, d=d, e=e).replace("; ", "\n   ")+"\n")
	result = lhs - rhs
	return signed_log(result)

def f6(x, y, *args):
	n = len(args[0])
	a = args[0][:n]
	b = args[1][:n]
	c = args[2][:n]
	d = args[3][:n]
	e = args[4][:n]

	lhs = 0
	rhs = 1

	for i in range(len(a)):
		# the result is the difference between the sum and the product of this many sine waves in 2d
		lhs += c[i]*np.sin(d[i]*x) + a[i]*np.sin(b[i]*y) + e[i]
		rhs *= a[i]*np.sin(b[i]*x) + c[i]*np.sin(d[i]*y) + e[i]

	print("f6; a = {a}; b = {b}; c = {c}; d = {d}; e = {e}".format(a=a, b=b, c=c, d=d, e=e).replace("; ", "\n   ")+"\n")
	result = lhs - rhs
	return signed_log(result)

def signed_log(x):
	return np.log(1+abs(x)) * np.sign(x)

def plot_deviation(f, *args):
	xlim = (-3, 3)
	ylim = (-3, 3)
	extent = list(xlim) + list(ylim)
	n_points = 1000
	xs = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0])/n_points)
	ys = np.arange(ylim[0], ylim[1], (ylim[1] - ylim[0])/n_points)

	X, Y = np.meshgrid(xs, ys[::-1])
	Z = f(X, Y, *args)

	zlim = (np.min(Z), np.max(Z))
	n_levels = 20
	zs = np.arange(zlim[0], zlim[1], (zlim[1] - zlim[0])/n_levels)
	widths = [0.5]*len(zs) + [1]
	zs = np.concatenate([zs, [0]])

	contours_misplaced = False
	matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

	if contours_misplaced:
		im = plt.imshow(Z, cmap="Spectral", extent=extent)
		cs = plt.contour(Z, extent=extent, colors="black", levels=zs, linewidths=widths)
	else:
		im = plt.imshow(Z, cmap="Spectral", extent=extent, origin="lower")
		cs = plt.contour(Z, extent=extent, colors="black", levels=zs, linewidths=widths, origin="lower")
	
	# plt.clabel(cs, inline=False, fmt='%1.1f', fontsize=np.nan, levels=[0])
	plt.colorbar(im)
	plt.show()

# plot_deviation(f2, random.uniform(-1, 1), random.uniform(-1, 1))
# plot_deviation(f3, random.uniform(-1, 1), random.uniform(-1, 1))
plot_deviation(f4,
	np.random.uniform(-1, 1, 4),
	np.random.uniform(-4, 4, 100),
	np.random.uniform(-1, 1, 100),
	np.random.uniform(-4, 4, 100),
	np.random.uniform(-1, 1, 100)
)
plot_deviation(f5,
	np.random.uniform(-2, 2, 4),
	np.random.normal(0, 2, 100),
	np.random.uniform(-2, 2, 100),
	np.random.normal(0, 2, 100),
	np.random.uniform(-1, 1, 100)
)
plot_deviation(f6,
	np.random.uniform(-2, 2, 7),
	np.random.normal(0, 2, 100),
	np.random.uniform(-2, 2, 100),
	np.random.normal(0, 2, 100),
	np.random.uniform(-1, 1, 100)
)


