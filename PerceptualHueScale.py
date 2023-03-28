# trying to make a function that shows how perceptual color maps to changes in wavelength or hue angle
# since it seems that hue angle (HueRotation.py) has way too much green and purple
# want perception of color category to vary more uniformly with angle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d


standard_hues = {
    "red": 0,
    "orange": 30,
    "yellow": 60,
    "green": 120,
    "blue": 240,
    "violet": 270,
    "red2": 360,
}

ryb_color_wheel_hues = {
    "red": 0,
    "orange": 60,
    "yellow": 120,
    "green": 180,
    "blue": 240,
    "violet": 300,
    "red2": 360,
}

experimental_hues = {
    "red": 0,
    "orange": 80,
    "yellow": 160,
    "green": 240,
    "blue": 280,
    "violet": 320,
    "red2": 360,
}

# desired_hues = ryb_color_wheel_hues  # still too much green and purple!
desired_hues = experimental_hues

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

spl_ys_stan_to_des_adjusted = spl_stan_to_des(xs)
spl_ys_des_to_stan_adjusted = spl_des_to_stan(xs)

spl_ys_stan_to_des = xs + spl_ys_stan_to_des_adjusted
spl_ys_des_to_stan = xs + spl_ys_des_to_stan_adjusted

# now we line up the two curves along lines perpendicular to y=x, so we can average the curves "toward" the y=x line (you can't just average them in their respective coordinate worlds because adding xs creates shear on the curve's shape)
# distance from y=x line of point (a,b) = (b-a)/sqrt(2)
# the x=y value at which that perpendicular descends to the y=x line from the point (a,b) is (a+b)/2
# then, given a new height d above the x=y line at effective x of x', (a,b) = (x' - d/sqrt(2), x' + d/sqrt(2))
effective_xs_stan_to_des = 1/2 * (xs + spl_ys_stan_to_des)
effective_xs_des_to_stan = 1/2 * (xs + spl_ys_des_to_stan)
heights_stan_to_des = 2**(-1/2) * (spl_ys_stan_to_des - xs)
heights_des_to_stan = 2**(-1/2) * (spl_ys_des_to_stan - xs)

# observe how the curve subtracting y=x differs from the heights off of the y=x line (shear)
# plt.plot(xs, spl_ys_stan_to_des_adjusted, c="b")
# plt.plot(xs, heights_stan_to_des, c="g")
# plt.plot(xs, spl_ys_des_to_stan_adjusted, c="r")
# plt.plot(xs, heights_des_to_stan, c="orange")
# plt.show()

# now want to match up the two curves in both "spaces", one where standard is the domain and the other where desired is the domain
# so the effective xs won't necessarily be the same because of the shear
# for each effective x (projecting onto y=x line), want to average the curves' heights at that point, thus getting a new height at this effective x, then transform that back into an (x,y) value using the formula for (a,b) above

# observe how the effective xs differ, in some places the reds are closer together than the blues and in others they are farther apart, and they rarely line up exactly, this is because of changes in the slopes causing different amounts of shear
# plt.scatter(effective_xs_stan_to_des, effective_xs_stan_to_des, c="b")
# plt.scatter(effective_xs_des_to_stan, effective_xs_des_to_stan, c="r")
# plt.show()

# could just linearly interpolate at each effective x in the whole set?
all_effective_xs = np.array(sorted(set(effective_xs_stan_to_des) | set(effective_xs_des_to_stan)))
# interpolate it at the same xs values that are evenly spaced, but use the (effective_x, height) points as the guide to interpolate from
height_by_effective_x_stan_to_des_func = interp1d(effective_xs_stan_to_des, heights_stan_to_des)
height_by_effective_x_des_to_stan_func = interp1d(effective_xs_des_to_stan, heights_des_to_stan)

# for here, define height as up above y=x from the perspective of the x axis being standard hue
heights_at_xs_stan_to_des = height_by_effective_x_stan_to_des_func(xs)
heights_at_xs_des_to_stan = -1 * height_by_effective_x_des_to_stan_func(xs)
average_heights_at_xs = 1/2 * (heights_at_xs_stan_to_des + heights_at_xs_des_to_stan)

# because of choosing the convention where positive height = des > stan (above the y=x line from the perspective of the x axis being standard hue)
new_stans = xs - average_heights_at_xs/(2**0.5)
new_dess = xs + average_heights_at_xs/(2**0.5)

get_average_height_at_x = lambda x: 1/2 * (height_by_effective_x_stan_to_des_func(x) + -1*height_by_effective_x_des_to_stan_func(x))
get_standard_hue_from_desired = lambda x: x - get_average_height_at_x(x)/(2**0.5)
get_desired_hue_from_standard = lambda x: x + get_average_height_at_x(x)/(2**0.5)
get_standard_hue_from_desired_01 = lambda x: 1/360 * get_standard_hue_from_desired(360*x)
get_desired_hue_from_standard_01 = lambda x: 1/360 * get_desired_hue_from_standard(360*x)

# done! now we have the function we want

if __name__ == "__main__":
    # plot in both "spaces"
    plt.subplot(2,2,1)
    plt.plot(xs, spl_ys_stan_to_des, c="b")
    plt.plot(xs, spl_ys_stan_to_des_adjusted, c="b", alpha=0.5)
    plt.plot(spl_ys_des_to_stan, xs, c="r")
    plt.plot(spl_ys_des_to_stan_adjusted, xs, c="r", alpha=0.5)
    plt.scatter(xs_stan_to_des, ys_stan_to_des, c="k")
    plt.xlabel("standard hue")
    plt.ylabel("desired hue")

    plt.subplot(2,2,2)
    plt.plot(spl_ys_stan_to_des, xs, c="b")
    plt.plot(spl_ys_stan_to_des_adjusted, xs, c="b", alpha=0.5)
    plt.plot(xs, spl_ys_des_to_stan, c="r")
    plt.plot(xs, spl_ys_des_to_stan_adjusted, c="r", alpha=0.5)
    plt.scatter(xs_des_to_stan, ys_des_to_stan, c="k")
    plt.xlabel("desired hue")
    plt.ylabel("standard hue")

    # in the lower row, plot the resulting function
    plt.subplot(2,2,3)
    plt.scatter(xs_stan_to_des, ys_stan_to_des, c="k")
    plt.plot(new_stans, new_dess, c="orange")
    plt.xlabel("standard hue")
    plt.ylabel("desired hue")

    plt.subplot(2,2,4)
    plt.scatter(xs_des_to_stan, ys_des_to_stan, c="k")
    plt.plot(new_dess, new_stans, c="orange")
    plt.xlabel("desired hue")
    plt.ylabel("standard hue")

    plt.show()

