# lim n->inf of (n!/n^n)^(1/n)

import math


def get_val(n):
    res = 1
    for i in range(n):
        term = ((n-i)/n) ** (1/n)
        res *= term
    return res


if __name__ == "__main__":
    for n in range(1000):
        print(get_val(n))
