import math
import msvcrt
import random


def u(n):
	return random.uniform(-n, n)

def get_function(n_var):
	# a = [u(100) for i in range(n_var)]
	a = [1.0/n_var for i in range(n_var)]
	b = [u(5) for i in range(n_var)]
	c = [u(100) for i in range(n_var)]
	s = [random.choice([math.sin, math.cos]) for i in range(n_var)]

	f = lambda args, i: a[i] * (s[i])(b[i] * args[i] + c[i])
	return lambda *args: sum([f(args, i) for i in range(len(args))])


print("Move through four-dimensional space using the numpad. (+,-) = w(7,3), x(6,4), y(8,2), z(9,1). Maximize f().")
f = get_function(4)
true_optimum = 1
w, x, y, z = (0.0, 0.0, 0.0, 0.0)
print(f(w, x, y, z))
optimum = f(w, x, y, z)
while True:
	if msvcrt.kbhit():
		o = ord(msvcrt.getch())
		if o == 224:
			# waste this value because it always shows up for some reason
			o = ord(msvcrt.getch())
		# print(o)
		if o == 9:
			raise KeyboardInterrupt
		elif o == 55: # northwest
		    w += 0.01
		elif o == 51: # southeast
		    w -= 0.01
		elif o == 54: # right
		    x += 0.01
		elif o == 52: # left
		    x -= 0.01
		elif o == 56: # up
		    y += 0.01
		elif o == 50: # down
		    y -= 0.01
		elif o == 57: # northeast
		    z += 0.01
		elif o == 49: # southwest
		    z -= 0.01
		print(f(w, x, y, z))
		if f(w, x, y, z) > optimum:
			print("new optimum! (log-distance from true optimum = {0})".format(math.log(true_optimum - optimum)))
			optimum = f(w, x, y, z)