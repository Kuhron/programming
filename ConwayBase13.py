import math
# import random


def number_to_base(n, b, min_power=-16):
    # returns dict of power:digit values e.g. (587, b=12) -> {2: 4, 1: 0, 0: 11}
    # eventually weird floating point math will mess up the expansion, using -16 as rule of thumb for area where this starts to happen
    if n == 0:
        return {0: 0}
    if n < 0:
        return -1 * number_to_base(-n, b)

    starting_power = math.floor(math.log(n,b))
    power = starting_power
    d = {}
    while n > 0 and power >= min_power:
        unit = b ** power
        digit = int(n // unit)
        d[power] = digit
        n %= unit
        power -= 1
    return d


def get_conway_value(d):
    # replace 10 with ., 11 with -, 12 with +
    powers = sorted(d.keys(), reverse=True)
    b = 10  # evaluate the resulting str in base 10
    s = ""
    for p in powers:
        digit = d[p]
        if digit in [11, 12]:  # correspond to - and +
            # got a sign, reset the string
            s = ""

        if digit == 10:
            c = "."
        elif digit == 11:
            c = "-"
        elif digit == 12:
            c = "+"
        elif digit > 12:
            raise ValueError(digit)
        else:
            c = str(digit)
        s += c
    if s == "":
        # invalid for now although it may be valid at greater expansion
        return 0
    if s.count(".") > 1:
        # invalid for now
        return 0
    if "+" not in s and "-" not in s:
        s = "+" + s
    assert s[0] in "+-"
    assert not any(x in "+-" for x in s[1:])  # should not happen because string would have been reset on new sign
    mult = -1 if s[0] == "-" else 1
    if "." not in s:
        s = s + ".0"
    # print("s = {}".format(s))
    pos_powers, neg_powers = s[1:].split(".")
    res = 0
    for power_i, digit in enumerate(pos_powers[::-1]):
        # power_i = 0 corresponds to ones place, so power is same as power_i
        power = power_i
        val = int(digit)
        res += val * (b ** power)
    for power_i, digit in enumerate(neg_powers):
        # power_i = 0 corresponds to first place after decimal, so power is -1
        power = -1 * (power_i + 1)
        val = int(digit)
        res += val * (b ** power)
    res *= mult
    return res



if __name__ == "__main__":
    assert number_to_base(587, 12) == {2:4, 1:0, 0:11}

    for n in range(1, 100):
        # print("n = {}".format(n))
        d = number_to_base(math.sqrt(n), 13, min_power=-32)
        print("sqrt({}) -> {}".format(n, get_conway_value(d)))

    for n in range(1, 100):
        d = number_to_base(math.exp(n), 13, min_power=-32)
        print("exp({}) -> {}".format(n, get_conway_value(d)))

