import sympy
import matplotlib.pyplot as plt

gaps = []
p0 = sympy.prime(1000000)
for _ in range(1000000):
    p1 = sympy.nextprime(p0)
    gaps.append(p1 - p0)
    p0 = p1

plt.scatter(range(len(gaps)), gaps)
plt.show()
