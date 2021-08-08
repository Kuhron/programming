import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy


def get_dna(length):
    return np.random.choice([0,1], length)


def print_dna(dna):
    # c = "01"
    c = ".|"
    s = ">" + "".join(c[x] for x in dna)
    print(s)


def transcribe_dna(dna, baseline_error_rate=1/1000):
    # make a copy of the dna with some error rate
    # bit replacement rate, bit insertion rate, bit deletion rate, jumping rate (jump to a random place in the sequence)
    flip_rate = baseline_error_rate * 1
    insert_rate = baseline_error_rate * 1
    delete_rate = baseline_error_rate * 1
    jump_back_rate = baseline_error_rate * 1
    jump_forward_rate = baseline_error_rate * 1/1.5
    prob_no_error = 1 - flip_rate - insert_rate - delete_rate - jump_back_rate - jump_forward_rate
    p = np.array([flip_rate, insert_rate, delete_rate, jump_back_rate, jump_forward_rate])
    p /= p.sum()
    choices = list("fidjJ")

    dna = deepcopy(dna)  # don't modify the original object
    res = []
    i = 0
    while i < len(dna):
        bit = dna[i]
        if random.random() < prob_no_error:
            res.append(bit)
            i += 1
            continue

        action = np.random.choice(choices, p=p)
        if action == "f":
            res.append(1 - bit)
            i += 1
        elif action == "i":
            res.append(random.choice([0,1]))
            res.append(bit)
            i += 1
        elif action == "d":
            pass
            i += 1
        elif action == "j":
            # jump backward
            i = random.randrange(0, i+1)
        elif action == "J":
            # jump forward
            i = random.randrange(i, len(dna))
            # for the jumps, always let i jump back to itself (make sure it's in the range) so you don't get empty range
        else:
            raise ValueError(f"unknown action {action}")
    return np.array(res)


def flip_bit(dna, index):
    dna = deepcopy(dna)  # don't modify the passed array
    dna[index] = 1 - dna[index]
    return dna


def plot_dna_as_path(dna, save=True, alpha=1):
    # +1 for 1, -1 for 0
    xs = np.array([1 if x == 1 else -1 for x in dna])
    cumsum = xs.cumsum()
    cumsum = cumsum - cumsum.mean()  # stupid np -= casting crap
    plt.plot(cumsum, alpha=alpha, c="b")
    if save:
        plt.savefig("DnaCumsum.png")
        plt.gcf().clear()


def plot_dnas_as_paths(dnas, save=True):
    alpha = 0.5
    for dna in dnas:
        plot_dna_as_path(dna, save=False, alpha=alpha)
    if save:
        plt.savefig("DnaCumsums.png")
        plt.gcf().clear()


if __name__ == "__main__":
    dna = get_dna(100)
    print_dna(dna)
    
    while len(dna) > 0:
        new_dna = transcribe_dna(dna)
        if len(new_dna) != len(dna) or (new_dna != dna).any():
            print_dna(new_dna)
            dna = new_dna
