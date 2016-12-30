import random

a = 0
b = 0
k = 100
n = 10000

for i in range(n):
	s = "".join([random.choice("()") for j in range(2*k)])
	try:
		eval(s)
		a += 1
	except SyntaxError:
		b += 1
	except TypeError:
		a += 1

r = a / (a + b)
print("Succeeded {0:.3f}% of the time".format(r * 100))