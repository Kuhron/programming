import numpy as np
import random


def get_dna(length):
    return np.random.choice([0,1], length)


def print_dna(dna):
    # c = "01"
    c = ".|"
    s = "".join(c[x] for x in dna)
    print(s)


def transcribe_dna(dna):
    # make a copy of the dna with some error rate
    # bit replacement rate, bit insertion rate, bit deletion rate, jumping rate (jump to a random place in the sequence)
    flip_rate = 1/1000
    insert_rate = 1/1000
    delete_rate = 1/1000
    jump_back_rate = 1/1000
    jump_forward_rate = 1/1500
    p = np.array([flip_rate, insert_rate, delete_rate, jump_back_rate, jump_forward_rate,
        1 - flip_rate - insert_rate - delete_rate - jump_back_rate - jump_forward_rate
    ])
    p /= p.sum()
    choices = list("fidjJn")  # n for none/normal

    res = []
    i = 0
    while i < len(dna):
        bit = dna[i]
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
        elif action == "n":
            res.append(bit)
            i += 1
        else:
            raise ValueError(f"unknown action {action}")
    return np.array(res)


def flip_bit(dna, index):
    dna[index] = 1 - dna[index]
    return dna


if __name__ == "__main__":
    dna = get_dna(100)
    print_dna(dna)
    
    while len(dna) > 0:
        new_dna = transcribe_dna(dna)
        if len(new_dna) != len(dna) or (new_dna != dna).any():
            print_dna(new_dna)
            dna = new_dna
