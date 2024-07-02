# my own investigation of this, want to try to prove some things myself
# here is the OEIS sequence for the number of sequences mod n: https://oeis.org/A015134


import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sympy


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
    # order = list(range(base))
    order = get_column_order_for_base(base)
    remaining_conditions = [[i, j] for i in order for j in order]
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
    seq_lens = [len(seq) - 2 for seq in seqs]  # len of sequence is actually number of pairs in it, = len(seq) - 2
    print("base {} has {} sequences".format(base, len(seqs)))
    print("lengths in order of sequence number: {}".format(seq_lens))
    print("lengths in sorted order:             {}".format(sorted(seq_lens)))
    tuple_to_seq_number = {}
    for i, seq in enumerate(seqs):
        print("seq #{}, len {} with factorization {}: {}".format(i, seq_lens[i], sympy.factorint(seq_lens[i]), seq))  
        pairs = get_conditions_in_sequence(seq)
        for pair in pairs:
            tuple_to_seq_number[tuple(pair)] = i

    print("\n"+ ("-"*40) +"\n")

    if base <= 50:
        show_table(base, tuple_to_seq_number)
    else:
        print("too big to show table")


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
    assert int(num) == num
    num = int(num)
    return num != 0 and ((num & (num - 1)) == 0)


def get_column_order_for_base(base):
    # if False: #is_power_of_2(base):
    #     return get_column_order_for_power_of_2(base)

    # prime power ordering
    factorization = sympy.factorint(base)
    if len(factorization) == 1:  # prime power, only one prime factor ignoring multiplicity
        p = list(factorization.keys())[0]
        k = factorization[p]
        if k == 1: return list(range(base))
        else:
            previous = get_column_order_for_base(p**(k-1))
            previous = [x*p for x in previous]
            rest = []
            for remainder in range(1, p):
                rest.extend([x + remainder for x in previous])
            return previous + rest

    # default ordering, contains good patterns of its own! do not dismiss!
    return list(range(base))


def get_column_order_for_power_of_2(base):
    assert is_power_of_2(base) and base >= 1
    base = int(base)
    if base == 1: return [0]  # this is the actual base case
    previous = get_column_order_for_power_of_2(base/2)
    previous = [x*2 for x in previous]

    # these are hacks, trying to see what the table looks like with different orders
    # if base == 2: rest = [1]
    # elif base == 4: rest = 
    # elif base == 8: rest = [7, 3, 5, 1]

    if True: # else:
        # once I figure out how this works, there should only be one "rest = " construction
        rest = [x + 1 for x in previous] # [x for x in range(base) if x not in previous][::-1]
    return previous + rest


def show_table(base, tuple_to_seq_number):
    show_only_coprime_pairs = input("show only coprime pairs? (y/[n]): ").strip().lower() == "y"
    # https://stackoverflow.com/questions/46663911/how-to-assign-specific-colors-to-specific-cells-in-a-matplotlib-table

    # if is_power_of_2(base):
    #     # even numbers first, so that you will see the upper left quarter is the same as the table for the previous power of 2
    #     # (previous sequences were all doubled)
    #     columns = get_column_order_for_power_of_2(base)
    # else:
    #     columns = [str(x) for x in range(base)]
    columns = get_column_order_for_base(base)
    rows = columns[:]

    n_seqs = max(tuple_to_seq_number.values())

    def f(r, c):
        # return random.choice(range(base))
        return tuple_to_seq_number[(r, c)]

    def color(x):
        assert 0 <= x <= n_seqs
        return plt.get_cmap("hsv")(x/(n_seqs+1))  # want 0 and 1 not to be the same color

    text_array = []
    color_array = []

    def effective_for_coprimality(n): return base if n == 0 else n
    def is_founder_tuple(x, y):
        if x == 0 and y == 0:
            return False
        x = effective_for_coprimality(x)
        y = effective_for_coprimality(y)
        if x == y:
            assert x != base
            return is_founder_tuple(x, base)
            # will inherit e.g. (3, 3)%12, from n=4, but not (11, 11)
            # and even though 10 is not prime, n=21 will not inherit (10, 10) from anywhere because 10 and 21 are coprime
        x_factors = set(sympy.primefactors(x))
        y_factors = set(sympy.primefactors(y))
        base_factors = set(sympy.primefactors(base))
        return x_factors & y_factors & base_factors == set()  # they share no prime factors

    for r_i in range(base):
        row_text_array = []
        row_color_array = []
        for c_i in range(base):
            r = rows[r_i]
            c = columns[c_i]
            v = f(r, c)
            row_text_array.append(" {} ".format(v))
            if show_only_coprime_pairs and not is_founder_tuple(r, c):
                row_color_array.append("#000000")
            else:
                row_color_array.append(color(v))
        text_array.append(row_text_array)
        color_array.append(row_color_array)

    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=text_array, cellColours=color_array, rowLabels=rows, colLabels=columns, loc='center')
    if base > 27:
        # doesn't fit on screen anymore
        table.auto_set_font_size(False)  # seriously there is an underscore here but not in set_fontsize()
        table.scale(0.5, 0.5)
        table.set_fontsize(4)
    else:
        table.auto_set_column_width(list(range(base)))
        # table.auto_set_row_height(list(range(base)))  # method doesn't exist
    plt.show()


if __name__ == "__main__":
    # add_to_lengths_file()  # best to do in background job, then comment out

    while True:
        try:
            base = int(input("n = ").strip())
        except ValueError:
            print("invalid int, try again")
            continue

        report_for_base(base)

