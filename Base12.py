import random
import time


def get_base_12_array(x):
    gg = x // 1728
    x -= gg * 1728
    g = x // 144
    x -= g * 144
    d = x // 12
    x -= d * 12
    return [gg, g, d, x]

def get_base_12_str(x):
    if x == 0: return "0"
    return get_base_12_str_from_array(get_base_12_array(x))

def get_base_12_str_from_array(arr):
    gg, g, d, x = arr
    s = ""
    # dec = "\u2D52"  # rotated 2 from Tifinagh script
    # el = "\u0190"  # rotated 3 from extended Latin
    dec = "X"
    el = "E"
    for val in [gg, g, d, x]:
        str_val = dec if val == 10 else el if val == 11 else str(val)
        s += str_val
    return remove_leading_zeros(s)

def get_base_10_str(x):
    return str(x)

def remove_leading_zeros(s):
    s = s.strip()
    n_zeros = 0
    while s[n_zeros] == "0":
        n_zeros += 1
    # return (" " * n_zeros) + s[n_zeros:]
    return s[n_zeros:]

def print_base_12(x):
    arr = get_base_12_array(x)
    gg, g, d, x = arr
    s = get_base_12_str_from_array(arr)
    s += " = "
    s += "" if gg == 0 else "{} great gross ".format(gg)
    s += "" if g == 0 else "{} gross ".format(g)
    s += "" if d == 0 else "{} dozen ".format(d)
    s += "" if x == 0 else str(x)
    print(s)

def get_number():
    n_digs = random.randint(2, 3)
    result = 0
    for i in range(n_digs):
        power = i
        min_digit = 1 if i == n_digs - 1 else 0  # no leading 0
        result += (random.randint(min_digit, 11)) * (12 ** power)
    return result

def test_once():
    n = get_number()
    str_n_10 = get_base_10_str(n)
    str_n_12 = get_base_12_str(n)
    starting_base = random.choice([10, 12])
    target_base = 10 if starting_base == 12 else 12
    starting_str = str_n_12 if starting_base == 12 else str_n_10
    target_str = str_n_10 if starting_base == 12 else str_n_12
    print("Convert {} from base {} to base {}".format(starting_str, starting_base, target_base))
    if starting_base == 12:
        print_base_12(n)
    inp = input()
    print()
    if inp.upper().strip() == target_str.upper().strip():
        print("Correct!")
    else:
        print(random.choice(["oops", "uh-oh", "doh", "bummer", "sucks to suck"]))
    time.sleep(1)
    print("the answer was {}".format(target_str))
    if target_base == 12:
        print_base_12(n)
    print()
    

if __name__ == "__main__":
    while True:
        test_once()
