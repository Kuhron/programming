import math
import sympy


def is_int(x):
    x += 0.5  # don't want to worry about floats being greater or less than the int by tiny amount
    return abs((x % 1) - 0.5) < 1e-6

def nearest_int(x):
    assert is_int(x)
    return round(x)

def conjecture_is_true(n):
    # n can be written as a prime plus 2*a^2 for natural a
    max_sq = math.floor(math.sqrt(n))
    sqs = [x**2 for x in range(1, max_sq + 1)]
    for sq in sqs:
        p_maybe = n - 2*sq
        if is_int(p_maybe):
            p_maybe = nearest_int(p_maybe)
            if sympy.isprime(p_maybe):
                return True
    return False

n = 9  # first odd composite
# n = 62600001  # got this far last time
while conjecture_is_true(n):
    if n % 100000 == 1: print(n)
    n += 2
    while sympy.isprime(n):
        n += 2

print(n)
