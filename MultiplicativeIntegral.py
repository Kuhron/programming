from __future__ import division

import math


def multiplicative_integral_discrete(f, a, b, n_steps):
	result = 1
	dx = (b - a) / n_steps
	for i in range(n_steps):
		x = a + (i + 1) * dx
		result *= f(x) ** dx
	return result


def test_constant_integrates_to_exponential():
	print(multiplicative_integral_discrete(lambda x: 2, 0, 4, 10))  # multiplicative integral over a constant function should always be base ** (b - a)
	print(multiplicative_integral_discrete(lambda x: 2, 0, 4, 1))
	print(multiplicative_integral_discrete(lambda x: 2, 3, 7, 50))


# print(multiplicative_integral_discrete(lambda x: x, 0, 1, 1000000))  # appears to be 1/e
# print(multiplicative_integral_discrete(lambda x: x, 0, 2, 1000000))  # appears to be 4/(e**2)
# print(multiplicative_integral_discrete(lambda x: x, 0, 3, 10000000))  # 27/(e**3)
# so multilplicative integral of (lambda x: x) from 0 to x is (x/e)**x

print(multiplicative_integral_discrete(lambda x: x, 1, 2, 1000000))  # just integral from 0 to 2, divided by integral from 0 to 1 (intuitive)

