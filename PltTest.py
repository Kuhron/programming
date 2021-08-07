# trying to get X11 forwarding to work on WSL
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

xs = np.linspace(0, 100, 101)
r = lambda: np.random.normal(0,10)
a, b, c = r(), r(), r()
f = lambda x: a*np.sin(x) + b*x + c
ys = f(xs)

plt.plot(xs, ys)
plt.show()
