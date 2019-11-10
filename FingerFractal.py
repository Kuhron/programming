import numpy as np
import matplotlib.pyplot as plt


# fractal hands on fingers
# thumb has opposite polarity
# left is 0, right is 1
# start with right hand
# read fingers left to right


def step(seq):
    # 0 is a left hand, so has thumb + 4f, becomes RLLLL
    # 1 is a right hand, so has 4f + thumb, becomes RRRRL
    res = []
    for x in seq:
        if x == 0:
            res += [1, 0, 0, 0, 0]
        elif x == 1:
            res += [1, 1, 1, 1, 0]
        else:
            raise
    return res


def plot(seq):
    seq = [-1 if x == 0 else x for x in seq]
    cs = np.array(seq).cumsum()
    plt.plot(cs)
    plt.show()


if __name__ == "__main__":
    seq = [1]  # right hand
    for _ in range(10):
        seq = step(seq)
    plot(seq)
