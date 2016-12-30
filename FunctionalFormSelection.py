import math
import random

import numpy as np
import matplotlib.pyplot as plt


# class Function:
# 	def __init__(self, parameters):
# 		self.parameters = parameters


# class Polynomial(Function):
# 	def __init__(self, parameters):
# 		super().__init__(parameters)
# 		self.function = lambda x_vec: []

augment = lambda x: max(1e-12, x) if x > 0 else min(-1e-12, x)

polynomial = lambda x, c, p: sum([c[i] *1.0/ augment(p[i]**2) * x**p[i] for i in range(len(c))])
exponential = lambda x, c, b: sum([c[i] *1.0/ augment(2**b[i]) * b[i]**x for i in range(len(c))])
sinusoidal = lambda x, c, b: sum([c[i] * math.sin(b[i]*x) for i in range(len(c))])
logarithmic = lambda x, c, b: sum([c[i] * math.log(1+abs(x), 1+abs(b[i])) for i in range(len(c))])
functions = [polynomial, exponential, sinusoidal, logarithmic]

# rand = lambda: random.paretovariate(1)
rand = lambda: random.normalvariate(0, 1)
random_vector = lambda n: [rand() for i in range(n)]
random_function_type = lambda: random.choice(functions)
random_function = lambda n: (lambda f, c, b: lambda x: f(x, c, b))(random_function_type(), random_vector(n), random_vector(n))

f1 = random_function(50)
f2 = random_function(50)
xs = np.arange(-2, 2, 0.0001)
ys = [f1(f2(x)) for x in xs]

plt.plot(xs, ys)
plt.show()