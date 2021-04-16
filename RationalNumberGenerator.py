import math

# positives only

def rational_number_tuple_generator():
    # go along diagonals, start each new one with 1/n and end with n/1
    diagonal_number = 1
    while True:
        # each diagonal
        for i in range(diagonal_number):
            p = 1 + i
            q = diagonal_number - i
            coprime = math.gcd(p,q) == 1
            if coprime:
                yield (p, q)
        diagonal_number += 1


def rational_number_generator():
    g = rational_number_tuple_generator()
    while True:
        p,q = next(g)
        yield p/q


if __name__ == "__main__":
    g = rational_number_generator()
    for i in range(100):
        print(next(g))
