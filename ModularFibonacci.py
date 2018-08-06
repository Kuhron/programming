def get_next_value(a, b, base):
    return (a + b) % base


def get_sequence_from_initial_conditions(a, b, base):
    seq = [a, b, get_next_value(a, b, base)]
    last = seq[-2:]
    while last != [a, b]:
        seq.append(get_next_value(last[0], last[1], base))
        last = seq[-2:]
    return seq


def get_conditions_in_sequence(seq):
    result = set()
    for a, b in zip(seq[:-1], seq[1:]):
        result.add((a, b))
    return sorted([list(x) for x in result])


def get_sequences_for_base(base):
    result = []
    remaining_conditions = [[i, j] for i in range(base) for j in range(base)]
    # all bases will have degenerate sequence of all zeros, but return it anyway
    while remaining_conditions != []:
        a, b = remaining_conditions[0]
        seq = get_sequence_from_initial_conditions(a, b, base)
        result.append(seq)
        sets = get_conditions_in_sequence(seq)
        for x in sets:
            if x in remaining_conditions:
                remaining_conditions.remove(x)
    return result


def get_sets_for_base(base):
    seqs = get_sequences_for_base(base)
    return [get_conditions_in_sequence(seq) for seq in seqs]


def get_n_sets(base):
    return len(get_sequences_for_base(base))


def report_for_base(base):
    seqs = get_sequences_for_base(base)
    print("base {} has {} sequences".format(base, len(seqs)))
    for seq in seqs:
        print(seq)
        print(get_conditions_in_sequence(seq))
        print()



if __name__ == "__main__":
    # base = 7
    # print(get_sequence_from_initial_conditions(0, 1, base))
    # report_for_base(base)

    with open("ModularFibonacciSetLengths.txt", "w") as f:
        for base in range(1, 200):
            f.write("{} : {}\n".format(base, get_n_sets(base)))

