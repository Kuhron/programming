import sympy

triangle = lambda n: sum(range(1, n+1))

n = 1
while sympy.divisor_count(triangle(n)) <= 500:
    n += 1

print(triangle(n))
