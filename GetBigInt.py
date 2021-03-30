import random
import numpy as np
import matplotlib.pyplot as plt


def get_big_int():
    a = 1
    stopping_probability = 0.001
    while True:
        if random.random() < stopping_probability:
            break
        if random.random() < 0.5:
            # add
            addend = random.randint(1, 100)
            # addend = get_big_int()
            a += addend
        else:
            # multiply
            multi = random.randint(2, 100)
            # multi = get_big_int()
            a *= multi
    return a


def get_big_ints(n):
    return [get_big_int() for i in range(n)]


def plot_hist():
    big_ints = get_big_ints(1000)
    arr = []
    for x in big_ints:
        # np.log fails for huge ints
        # lx = np.log(x)
        lx = len(str(x))
        print("x has {} digits".format(lx))
        arr.append(lx)
    plt.hist(arr)
    plt.show()


def get_digit():
    return random.choice("0123456789")


def get_big_int_str_method(n_digits):
    return int("".join(get_digit() for i in range(n_digits)))


if __name__ == "__main__":
    # for i in range(100):
    #     print(get_big_int())
    # plot_hist()

    # can we random.seed on huge ints?
    for i in range(1000):
        x = get_big_int_str_method(n_digits=100000)
        random.seed(x)
        print("random seeded successfully, .random() gives {}".format(random.random()))
