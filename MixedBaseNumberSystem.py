import random
import math


def get_syllable():
    onsets = "ptkmnshywlr "
    vowels = "aeiou"
    codas = "mnslryw "
    o = random.choice(onsets)
    v = random.choice(vowels)
    c = random.choice(codas)
    return (o + v + c).replace(" ","")


def get_morpheme(n_syllables=None, exclusion_list=None):
    if n_syllables is None:
        n_syllables = random.choice([1]*4 + [2]*2 + [3] * 1)
    while True:
        res = "".join(get_syllable() for i in range(n_syllables))
        if exclusion_list is None or res not in exclusion_list:
            return res


def get_unique_morphemes(n_morphemes):
    res = []
    while len(res) < n_morphemes:
        m = get_morpheme()
        if m not in res:
            res.append(m)
    return res


def get_small_int():
    min_n = 2
    n = min_n
    while random.random() < 0.9:
        n += 1
    return n


def get_bases():
    n_bases = random.randint(3, 5)
    bases = [1, get_small_int()]
    for i in range(n_bases-1):
        if bases[-1] <= 40 and random.random() < 0.5:
            next_base = bases[-1] + get_small_int()
        else:
            next_base = bases[-1] * get_small_int()
        bases.append(next_base)

    bases = add_large_powers(bases)
    return bases


def add_large_powers(bases):
    ratios = get_ratios(bases)
    r = main_ratio_for_large_numbers = random.choice([x for x in ratios if x != 1])
    b = bases[-1]
    n_large_powers = random.randint(1, 5)
    large_powers = [b * (r ** (i+1)) for i in range(n_large_powers)]
    return bases + large_powers


def express_number_in_bases(n, bases):
    res = [0 for b in bases]
    original_n = n
    for i in range(len(res)):
        # go from biggest to smallest base
        base_index = -(1+i)
        b = bases[base_index]
        place_value, n = divmod(n, b)
        res[base_index] = place_value
    assert convert_place_values_to_number(res, bases) == original_n
    return res


def convert_place_values_to_number(vec, bases):
    return sum(a*b for a,b in zip(vec, bases))


def report_arithmetic_for_number_in_bases(n, bases):
    vec = express_number_in_bases(n, bases)
    monomial_lst = []
    for a, b in zip(vec, bases):
        if a != 0:
            if b == 1:
                monomial = str(a)
            elif a == 1:
                monomial = str(b)
            else:
                a_str = report_arithmetic_for_number_in_bases(a, bases)
                monomial = f"({a_str})*{b}"
            monomial_lst.append(monomial)
    return " + ".join(monomial_lst[::-1])


def get_ratios(bases):
    ratios = [b//a for a,b in zip(bases[:-1], bases[1:])]
    return ratios


def get_morphemes_for_number_system(bases):
    # need a morpheme for each base greater than one
    # need a morpheme for zero
    # need a morpheme for each counting number up to the biggest one you'll encounter (either when going to the first base OR when counting number of some base until the next base)
    ratios = get_ratios(bases)
    # max_counting_number = max(ratios) - 1  # overshoots, e.g. has words all the way up to 20 when you could use sub-base of 3 to express them
    max_counting_number = bases[1] - 1
    counting_numbers = list(range(0, max_counting_number+1))
    base_numbers = bases[1:]

    numbers_to_name = counting_numbers + base_numbers
    morphemes = get_unique_morphemes(len(numbers_to_name))
    d = {n: m for n, m in zip(numbers_to_name, morphemes)}

    if random.random() < 0.5:
        and_morpheme = get_morpheme(n_syllables=1, exclusion_list=morphemes)
        d["and"] = and_morpheme
    else:
        d["and"] = None
    return d


def convert_number_to_language(n, bases, morphemes):
    if n in morphemes:
        return morphemes[n]
    vec = express_number_in_bases(n, bases)
    monomial_lst = []
    for a, b in zip(vec, bases):
        if a != 0:
            if b == 1:
                monomial = convert_number_to_language(a, bases, morphemes)
            elif a == 1:
                monomial = convert_number_to_language(b, bases, morphemes)
            else:
                a_str = convert_number_to_language(a, bases, morphemes)
                b_str = convert_number_to_language(b, bases, morphemes)
                monomial = f"{a_str} {b_str}"
            monomial_lst.append(monomial)
    and_word = morphemes["and"]
    delim = (" " + and_word + " ") if and_word is not None else " "
    return delim.join(monomial_lst[::-1])



if __name__ == "__main__":
    bases = get_bases()
    print("bases:", bases, "of ratios", get_ratios(bases))
    morphemes = get_morphemes_for_number_system(bases)
    print(morphemes)

    for n in [100, 1729, 43560, 131072]:
        print(f"{n} =", express_number_in_bases(n, bases))
        print(f"{n} =", report_arithmetic_for_number_in_bases(n, bases))
        print(f"{n} =", convert_number_to_language(n, bases, morphemes))
    print("----")

    for n in range(0, 100):
        print(f"{n} =", convert_number_to_language(n, bases, morphemes))
