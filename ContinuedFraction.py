import math

FLOAT_ERROR = (0.1 + 0.2) % 0.1
# print(FLOAT_ERROR)

def get_continued_fraction(x, n_terms):
    if n_terms <= 0:
        return []
    a = int(x/1)
    x -= a
    rest = get_continued_fraction(1/x, n_terms-1) if x != 0 else []
    return [int(a)] + rest

def get_neg1_continued_fraction(x, n_terms):
    if n_terms <= 0:
        return []
    a = int(x/1)
    x -= (a+1)
    # subtract a + 1 to get negative number of smallest magnitude
    # remainder is now rem - 1, for recursion
    # e.g. pi would normally be [3] + f(1/0.14159)
    # but here is [4] + f(-1/-0.85841)
    rest = get_neg1_continued_fraction(-1/x, n_terms-1) if x != 0 else []
    return [int(a+1)] + rest

# later can generalize to other numerator values


if __name__ == "__main__":
    rt = math.sqrt
    pi = math.pi
    e = math.exp(1)
    golden_ratio = (1+math.sqrt(5))/2
    for i, x in enumerate([
        # pi, e, golden_ratio, rt(2), rt(3), 
        # rt(5), rt(6), rt(7), rt(pi), pi**2 / 6, 
        # pi + e, pi - e, pi * e, pi ** e, e ** pi,
        # pi / e, e / pi, pi ** pi, e ** e,
        pi, e ** pi,
    ]):
        print(i, ":", x)
        print(get_continued_fraction(x, 20))
        print(get_neg1_continued_fraction(x, 20))
        print()
