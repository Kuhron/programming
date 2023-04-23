import numpy as np
import matplotlib.pyplot as plt

from ChaosMathUtil import get_trajectory_of_differential_equation

# making a simple example of a differential equation and corresponding difference equation
# making the motion vector exactly perpendicular only works for dt=0
# so I create an example showing that doing this with small but positive dt makes it spiral outwards
# and then I create a modified difference equation that corrects for this.
# I wonder if this kind of correction could be generalized? For the circle, we can "jump ahead" and know where it's going to be, so we can make an exact correction, but might not be possible in general. It probably relies on being able to integrate in closed form or something.

r = lambda *xs: sum(x**2 for x in xs)**0.5

# here we just take the differential equation at face value
dx = lambda x, y, dt: dt * ( -y * 2*np.pi )
dy = lambda x, y, dt: dt * ( x * 2*np.pi )
x0, y0 = np.random.random(2)

xs1, ys1 = get_trajectory_of_differential_equation([dx, dy], [x0, y0], 1, 0.01, 1e6)

plt.scatter(xs1, ys1, alpha=0.7)

# now we adjust the angle of the motion so it stays on the correct circle
dx = lambda x, y, dt: x * np.cos(2*np.pi*dt) - y * np.sin(2*np.pi*dt) - x
dy = lambda x, y, dt: y * np.cos(2*np.pi*dt) + x * np.sin(2*np.pi*dt) - y

xs2, ys2 = get_trajectory_of_differential_equation([dx, dy], [x0, y0], 1, 0.01, 1e6)

plt.scatter(xs2, ys2, alpha=0.7)
plt.scatter([x0], [y0], c="k", marker="x")
plt.savefig("Images/DECircle.png")
plt.gcf().clear()


