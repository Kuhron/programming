import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from FunctionOfBits import plus_minus_cumsum


def get_dna(length):
    return np.random.choice([0,1], length)


def print_dna(dna):
    print(get_dna_str(dna))


def get_dna_str(dna):
    # c = "01"
    c = ".|"
    s = ">" + "".join(c[x] for x in dna)
    return s


def transcribe_dna_iterative(dna, baseline_error_rate=1/1000):
    # make a copy of the dna with some error rate
    # bit replacement rate, bit insertion rate, bit deletion rate, jumping rate (jump to a random place in the sequence)
    flip_rate = baseline_error_rate * 1
    insert_rate = baseline_error_rate * 1
    delete_rate = baseline_error_rate * 1
    jump_back_rate = baseline_error_rate * 1
    jump_forward_rate = baseline_error_rate * 1/1.5
    # prob_no_error = 1 - flip_rate - insert_rate - delete_rate - jump_back_rate - jump_forward_rate
    prob_no_error = 1 - flip_rate - insert_rate - delete_rate
    # p = np.array([flip_rate, insert_rate, delete_rate, jump_back_rate, jump_forward_rate])
    p = np.array([flip_rate, insert_rate, delete_rate])
    p /= p.sum()
    # choices = list("fidjJ")
    choices = list("fid")

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


def transcribe_dna_using_arrays(dna, baseline_error_rate=1/1000):
    # see if this is faster than iteratively making the list myself
    n, = dna.shape
    flip_rate = baseline_error_rate * 1
    insert_rate = baseline_error_rate * 1
    delete_rate = baseline_error_rate * 1
    prob_no_error = 1 - flip_rate - insert_rate - delete_rate
    p = np.array([prob_no_error, flip_rate, insert_rate, delete_rate])
    p /= p.sum()
    mask = np.random.choice(range(len(p)), (n,))  # which action to do at each bit
    flip_mask = mask == 1
    delete_mask = mask == 3
    insert_indices = (mask == 2).nonzero()
    new_dna = np.array([x for x in dna])
    new_dna[flip_mask] = 1 - new_dna[flip_mask]
    new_dna[delete_mask] = -1
    insertions = np.random.choice([0, 1], (len(insert_indices),))
    # segs = [new_dna[:insertion_indices[0]]] + [new_dna[insertion_indices[i] : insertion_indices[i+1]] for i in range(len(insertion_indices)-1)] + [new_dna[insertion_indices[-1]]]
    # for i in range(1, len(segs)):
    #     # place insertion at beginning of this segment
    #     segs[i] = np.insert(segs[i], 0, insertions[i])
    for i in range(len(insert_indices)):
        insert_index = insert_indices[i] + i
        # i will also be the number of additional things in new_dna now, so we offset index with it
        new_dna = np.insert(new_dna, insert_index, insertions[i])
    new_dna = new_dna[new_dna != -1]  # get rid of deleted bases
    return new_dna


def flip_bit(dna, index):
    dna = deepcopy(dna)  # don't modify the passed array
    dna[index] = 1 - dna[index]
    return dna


def plot_dna_as_path(dna, path_func, save=True, alpha=1):
    path = path_func(dna)
    plt.plot(path, alpha=alpha, c="b")
    if save:
        plt.savefig("DnaPath.png")
        plt.gcf().clear()


def plot_dnas_as_paths(dnas, path_func, save=True):
    alpha = 0.5
    for dna in dnas:
        plot_dna_as_path(dna, path_func, save=False, alpha=alpha)
    if save:
        plt.savefig("DnaPaths.png")
        plt.gcf().clear()


if __name__ == "__main__":
    dna = get_dna(100)
    # print_dna(dna)

    lengths = []
    iterations = 0

    # somehow these give different results?
    # transcribe_dna = transcribe_dna_iterative
    transcribe_dna = transcribe_dna_using_arrays

    while len(dna) > 0:
        new_dna = transcribe_dna(dna)
        # print_dna(new_dna)
        dna = new_dna
        lengths.append(len(dna))
        if iterations % 1000 == 0:
            print(f"{iterations} iterations complete")
        iterations += 1
    plt.plot(lengths)
    plt.show()
