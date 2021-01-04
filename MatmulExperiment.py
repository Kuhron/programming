# just tried a random math experiment idea to see if i would find something interesting, and i did! what is the 0.631 number where infinite magnitude starts to show up in the end behavior? haven't tried this with any other matrices/functions of xy
# mostly copied stuff from ~/python_history because i need to sleep
import numpy as np
xy = np.random.rand(2)
xy
x,y = xy
m = np.array([[x+y,x-y],[-x+y,-x-y]])
m
np.matmul(m,xy)
get_m2 = lambda x,y: np.array([[x+y,x-y],[-x+y,-x-y]])
get_m = lambda xy: get_m2(*xy)

# note that the iterated function due to this matrix multiplication is:
# (further note that matmul 2x2 by (2,) treats the latter as column vector as it should, so np.matmul(np.array([[1,2],[3,4]),np.array([0,1])) is np.array([2,4])
# x1 = (x+y)*x + (x-y)*y = x**2 +2*x*y -y**2
# y1 = (-x+y)*x + (-x-y)*y = -x**2 -y**2
# so another way to state the same operation is as something like
# x1 =    1 y**1 y**2
#    1    0    0   -1
# x**1    0    2    0
# x**2    1    0    0
#
# y1 =    1 y**1 y**2
#    1    0    0   -1
# x**1    0    0    0
# x**2   -1    0    0

get_m(xy)
get_m(xy)
xy
f = lambda xy: np.matmul(get_m(xy), xy)
f(xy)
f(f(xy))
for i in range(10):
 xy = f(xy)
 print(xy)
def iterate(f, x, n):
 res = []
 res.append(x)
 for i in range(n):
  x = f(x)
  res.append(x)
 return res
iterate(f,xy,10)
xy
iterate(f, np.random.rand(2), 10)
import matplotlib.pyplot as plt
def plot_iterate_2d(f, xy, n):
 iterations = iterate(f, xy, n)
 xs = [xy[0] for xy in iterations]
 ys = [xy[1] for xy in iterations]
 plt.plot(xs, ys)
 plt.scatter(xs[0], ys[0], c="red")
 plt.show()
# plot_iterate_2d(f, np.random.rand(2), 10)
# plot_iterate_2d(f, np.random.uniform(-0.5, 0.5, (2,)), 10)
# plot_iterate_2d(f, np.random.uniform(-0.1, 0.1, (2,)), 10)
k = np.sqrt(2)/2
# plot_iterate_2d(f, np.random.uniform(-k, k, (2,)), 10)
# plot_iterate_2d(f, np.random.uniform(-k, k, (2,)), 10)
def describe_end_behavior(f, xy):
 order_of_mag = 40
 iterations = iterate(f, xy, 100)
 for new_xy in iterations:
  mag = np.linalg.norm(xy)
  log10 = np.log10(mag)
  if log10 > order_of_mag:
   return "inf"
  elif log10 < -order_of_mag:
   return "zero"
 return "neither"
describe_end_behavior(f, np.random.uniform(-k, k, (2,)))
k = 1
describe_end_behavior(f, np.random.uniform(-k, k, (2,)))
def describe_end_behavior(f, xy):
 order_of_mag = 40
 iterations = iterate(f, xy, 100)
 for new_xy in iterations:
  mag = np.linalg.norm(new_xy)
  log10 = np.log10(mag)
  if log10 > order_of_mag:
   return "inf"
  elif log10 < -order_of_mag:
   return "zero"
 return "neither"
describe_end_behavior(f, np.random.uniform(-k, k, (2,)))
get_xy_with_mag = lambda mag: mag * get_xy_with_mag_1()
def get_xy_with_mag_1():
 theta = np.random.uniform(0, 2*np.pi)
 y = np.sin(theta)
 x = np.cos(theta)
 return np.array([x,y])
get_xy_with_mag(3)
get_xy_with_mag(100)
describe_end_behavior(f, get_xy_with_mag(1))
describe_end_behavior(f, get_xy_with_mag(0.1))
describe_end_behavior(f, get_xy_with_mag(0.5))
describe_end_behavior(f, get_xy_with_mag(0.7))
describe_end_behavior(f, get_xy_with_mag(0.8))
end_behaviors = {}

mags = np.linspace(0.6, 1, 100)  # shows curve from ~0.63 to 1, where inf_prop goes from 0 to 1
# mags = np.linspace(0.630930, 0.630931, 20)  # focusing on the onset of the curve, trying to find out its value, is in range (0.630930, 0.630931), seems close to 0.63093058, no good closed forms on Wolfram Alpha
for mag_i, mag in enumerate(mags):
 print("mag {}/{}".format(mag_i, len(mags)))
 results = [describe_end_behavior(f, get_xy_with_mag(mag)) for i in range(1000)]
 zeros = results.count("zero")
 infs = results.count("inf")
 neithers = results.count("neither")
 d = {"inf_prop": infs / len(results), "zero_prop": zeros/len(results), "neither_prop":neithers/len(results)}
 end_behaviors[mag] = d
end_behaviors
ys = [end_behaviors[x]["inf_prop"] for x in mags]
plt.plot(mags, ys)
plt.show()

# checking if neither_prop is ever any appreciable amount (it's not)
# neither_props = [end_behaviors[x]["neither_prop"] for x in mags]
# plt.plot(mags, neither_props)
# plt.show()
