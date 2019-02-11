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
            print("n = {}".format(base))
            f.write("{}{}{}\n".format(base, delim, get_n_sets(base)))
            base += 1


def is_power_of_2(num):
    # http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
    return num != 0 and ((num & (num - 1)) == 0)


def show_table(base, tuple_to_seq_number):
    # https://stackoverflow.com/questions/46663911/how-to-assign-specific-colors-to-specific-cells-in-a-matplotlib-table

    # if is_power_of_2(base):
    #     # even numbers first, so that you will see the upper left quarter is the same as the table for the previous power of 2
    #     # (previous sequences were all doubled)
    #     # never mind this isn't true
    #     columns = sorted(range(base), key=lambda x: (x % 2, x))
    if True: # else:
        columns = [str(x) for x in range(base)]
    rows = columns[:]

    def f(r, c):
        # return random.choice(range(base))
        return tuple_to_seq_number[(r, c)]

    def color(x):
        assert 0 <= x <= base
        return cm.get_cmap("Spectral")(x/base)

    text_array = []
    color_array = []
    for r in range(base):
        row_text_array = []
        row_color_array = []
        for c in range(base):
            v = f(r, c)
            row_text_array.append(" {} ".format(v))
            row_color_array.append(color(v))
        text_array.append(row_text_array)
        color_array.append(row_color_array)

    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=text_array, cellColours=color_array, rowLabels=rows, colLabels=columns, loc='center')
    table.auto_set_column_width(list(range(base)))
    plt.show()


if __name__ == "__main__":
    add_to_lengths_file()

    for base in [4, 8, 16]:
        report_for_base(base)

