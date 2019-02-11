import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
    tuple_to_seq_number = {}
    for i, seq in enumerate(seqs):
        print("seq #{}".format(i))
        print(seq)
        pairs = get_conditions_in_sequence(seq)
        for pair in pairs:
            tuple_to_seq_number[tuple(pair)] = i
        print()

    show_table(base, tuple_to_seq_number)


def add_to_lengths_file():
    fp = "ModularFibonacciSetLengths.txt"
    delim = " : "
    with open(fp) as f:
        lines = f.readlines()
    lines = [x.strip().split(delim) for x in lines]
    xs = [int(item[0]) for item in lines]
    # ys = [int(item[1]) for item in lines]
    assert xs == sorted(set(xs)), "values of n in the lengths file are messed up"

    base = max(xs) + 1

    with open(fp, "a") as f:
        while True:
            f.write("{}{}{}\n".format(base, delim, get_n_sets(base)))
            base += 1


def test_table(base, tuple_to_seq_number):
    # https://stackoverflow.com/questions/46663911/how-to-assign-specific-colors-to-specific-cells-in-a-matplotlib-table
    columns = [str(x) for x in range(base)]
    rows = columns[:]

    def f(r, c):
        # return random.choice(range(base))

    def color(x):
        assert 0 <= x <= n
        return cm.get_cmap("Spectral")(x/n)

    text_array = []
    color_array = []
    for r in range(n):
        row_text_array = []
        row_color_array = []
        for c in range(n):
            v = f(r, c)
            row_text_array.append(str(v))
            row_color_array.append(color(v))
        text_array.append(row_text_array)
        color_array.append(row_color_array)

    plt.axis('tight')
    plt.axis('off')
    plt.table(cellText=text_array, cellColours=color_array, rowLabels=rows, colLabels=columns, loc='center')
    plt.show()


if __name__ == "__main__":
    base = 4
    report_for_base(base)

    # add_to_lengths_file()
    # test_table()
