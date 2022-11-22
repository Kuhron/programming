import random


def step(seq):
    res = []
    digit = None
    count = 0
    for i in range(len(seq) + 1):
        if i == len(seq):
            # terminate
            res += [count, digit]
            break
        else:
            x = seq[i]

        if digit is None:
            digit = x
            assert count == 0

        if x == digit:
            count += 1
        else:
            res += [count, digit]
            digit = x
            count = 1
    return res


def print_seq(seq):
    s = ""
    for x in seq:
        if 0 <= x <= 9:
            s += str(x)
        else:
            s += "(" + str(x) + ")"
    print(s)


if __name__ == "__main__":
    seq = [1]
    last_len = None
    for i in range(2000):
        # print_seq(seq)
        n = len(seq)
        r = None if last_len is None else n/last_len
        print(i, n, r)
        last_len = n
        seq = step(seq)

