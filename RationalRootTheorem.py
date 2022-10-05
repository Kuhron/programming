# polynomial a_n * x^n + ... + a_0 * x^0
# any rational solution p/q must satisfy
# p is a factor of a_0
# q is a factor of a_n


import random


def get_divisors(n):
    n = abs(n)
    res = [i for i in range(1, n+1) if n % i == 0]
    res = res + [-x for x in res]
    return res


def get_random_coefficients():
    n_coeffs = 3
    while random.random() < 1-0.2:
        n_coeffs += 1
    res = []
    for i in range(n_coeffs):
        a = 0
        while random.random() < 1-0.1:
            a += random.normalvariate(0, 5)
        a = round(a)
        res.append(a)
    while res[0] == 0:
        res = res[1:]
    while res[-1] == 0:
        res = res[:-1]
    if len(res) == 0:
        # try again
        return get_random_coefficients()
    return res


def evaluate_polynomial(coeffs, x):
    # power is index
    res = 0
    for i, a in enumerate(coeffs):
        res += a * (x ** i)
    return res


def coprime(a, b):
    a = abs(a)
    b = abs(b)
    for i in range(2, min(a, b)+1):
        if a % i == 0 and b % i == 0:
            return False
    return True


if __name__ == "__main__":
    coeffs = get_random_coefficients()
    print("coeffs:", coeffs)
    a0 = coeffs[0]
    an = coeffs[-1]
    divs_0 = get_divisors(a0)
    divs_n = get_divisors(an)
    rational_roots = []
    tried_xs = set()
    for p in divs_0:
        for q in divs_n:
            if not coprime(p, q):
                # print(f"{p} and {q} are not coprime")
                continue
            x = p / q
            if x in tried_xs:
                continue
            else:
                tried_xs.add(x)
            y = evaluate_polynomial(coeffs, x)
            print(f"{p}/{q} -> {y}")
            if y == 0:
                print("rational root found!")
                rational_roots.append((x, p, q))
    print("rational roots:", rational_roots)
