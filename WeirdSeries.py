# weird series from a math post I saw


def king_iterator():
    k = 1
    while True:
        for pair in get_king(k):
            yield pair
        k += 1


def get_king(k, starting_index=1):
    # (i, k), (k, i), and (k, k)
    for i in range(starting_index, k):  # only through k-1
        yield (i, k)
        yield (k, i)
    yield (k, k)


def term(m, n):
    return (n**2 - m**2) / (n**2 + m**2)**2


def get_partial_sums(max_k):
    # it's done in kings
    # new value of k means you have to do (m,n) in (1:k-1, k), (k, 1:k-1), and (k, k)
    total = 0
    for k in range(1, max_k + 1):
        for m, n in get_king(k, starting_index=1):
            total += term(m, n)
        yield k, total


if __name__ == "__main__":
    max_k = 1000000
    for k, partial_sum in get_partial_sums(max_k):
        if k % 1000 == 0:
            print(k, partial_sum)
