import sympy

print(sum(n for n in range(2, 2000000) if sympy.isprime(n)))
