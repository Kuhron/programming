# trying to make a function that shows how perceptual color maps to changes in wavelength or hue angle
# since it seems that hue angle (HueRotation.py) has way too much green and purple
# want perception of color category to vary more uniformly with angle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


standard_hues = {
    "red": 0,
    "orange": 30,
    "yellow": 60,
    "green": 120,
    "blue": 240,
    "violet": 270,
    "red2": 360,
}

desired_hues = {
    "red": 0,
    "orange": 60,
    "yellow": 120,
    "green": 180,
    "blue": 240,
    "violet": 300,
    "red2": 360,
}

# want a function mapping between these, one to one, use spline or something, ideally it's smooth, same derivative on both ends, since this loops around

# make a mapping from real to desired, subtract the linear trend and then model as a periodic spline
standard_items = sorted(standard_hues.items(), key=lambda kv: kv[1])
colors = [c for c,x in standard_items]
xs_stan_to_des = [x for c,x in standard_items]
ys_stan_to_des = [desired_hues[c] for c in colors]
ys_stan_to_des_adjusted = [y-x for x,y in zip(xs_stan_to_des, ys_stan_to_des)]

xs_des_to_stan = ys_stan_to_des
ys_des_to_stan = xs_stan_to_des
ys_des_to_stan_adjusted = [y-x for x,y in zip(xs_des_to_stan, ys_des_to_stan)]

# also try fitting spline on the inverse and see how different they are
# maybe do something like average them compromise between the fits? would need to do this both in one's "space" and the other (i.e. invert one and keep the other, and average those, then do the same the other way around to get two new curves, then maybe iterate and average those together again? how far to do this? to some tolerance level of their abs difference doesn't exceed some value?)
# as long as the result is one-to-one and doesn't have any huge slopes, we should be okay

spl_stan_to_des = CubicSpline(xs_stan_to_des, ys_stan_to_des_adjusted, bc_type="periodic")
spl_des_to_stan = CubicSpline(xs_des_to_stan, ys_des_to_stan_adjusted, bc_type="periodic")

xs = np.linspace(0, 360, 360*60)

plt.plot(xs, spl_stan_to_des(xs), c="b")
plt.plot(xs, -1*spl_des_to_stan(xs), c="r")
plt.show()

spl_ys_stan_to_des = spl_stan_to_des(xs)
spl_ys_des_to_stan = spl_des_to_stan(xs)

# avg_ys_stan_to_des = ?
# avg_ys_des_to_stan = ?

# plot in both "spaces"
plt.subplot(2,2,1)
plt.plot(xs, spl_ys_stan_to_des, c="b")
plt.plot(spl_ys_des_to_stan, xs, c="r")
plt.scatter(xs_stan_to_des, ys_stan_to_des, c="k")
plt.xlabel("standard hue")
plt.ylabel("desired hue")

plt.subplot(2,2,2)
plt.plot(spl_ys_stan_to_des, xs, c="b")
plt.plot(xs, spl_ys_des_to_stan, c="r")
plt.scatter(xs_des_to_stan, ys_des_to_stan, c="k")
plt.xlabel("desired hue")
plt.ylabel("standard hue")

# in the lower row, plot the resulting function
plt.subplot(2,2,3)
# plt.plot(xs, avg_ys_stan_to_des, c="b")
# plt.plot(avg_ys_des_to_stan, xs, c="r")
plt.scatter(xs_stan_to_des, ys_stan_to_des, c="k")
plt.xlabel("standard hue")
plt.ylabel("desired hue")

plt.subplot(2,2,4)
# plt.plot(avg_ys_stan_to_des, xs, c="b")
# plt.plot(xs, avg_ys_des_to_stan, c="r")
plt.scatter(xs_des_to_stan, ys_des_to_stan, c="k")
plt.xlabel("desired hue")
plt.ylabel("standard hue")

plt.show()

