from datetime import date
import random
import matplotlib.pyplot as plt
import math
import re


def get_random_substring(s, min_len=None, max_len=None):
    if min_len is None:
        min_len = 1
    assert min_len >= 1, min_len
    if max_len is None:
        max_len = len(s)
    assert max_len >= 1, max_len
    assert min_len <= max_len, (min_len, max_len)

    length_to_use = random.randrange(min_len, max_len+1)
    n = len(s)
    max_start = n - length_to_use
    start = random.randrange(max_start)
    end = start + length_to_use
    assert 0 <= start, start
    assert end <= n - 1, (end, n)
    assert start < end, (start, end)
    return s[start : end]


def get_day_number():
    d0 = date(1970, 1, 1)
    d1 = date.today()
    return (d1 - d0).days


def plot_trajectory_over_days(s, n_days):
    vals = []
    for i in range(n_days):
        val = get_rand_val(s, day_number_fudge=i)
        vals.append(val)
    plt.scatter(range(n_days), vals)
    plt.show()


def get_rand_val(s, day_number_fudge=0):
    # seed values, one for the day and the rest from the string
    # day_number_fudge is just for checking that a string's trajectory over values is sufficiently random from day to day
    day_num = get_day_number() + day_number_fudge
    l = len(s)
    l_no_vowels = len(re.sub("[aoeui]", "", s))
    l_no_lower = len(re.sub("[a-z]", "", s))
    l_no_upper = len(re.sub("[A-Z]", "", s))
    l_no_AM = len(re.sub("[A-Ma-m]", "", s))
    l_no_NZ = len(re.sub("[N-Zn-z]", "", s))
    first_code = ord(s[0])
    last_code = ord(s[-1])
    H2,I2,J2,K2,L2,M2,N2,O2 = l,l_no_vowels,l_no_lower,l_no_upper,l_no_AM,l_no_NZ,first_code,last_code

    # now mush stuff together
    day_mod = day_num % (17.17 * math.pi)
    alt_sum_prod_1 = H2 * I2 + J2 * K2 + L2 * M2 + N2 * O2
    alt_sum_prod_2 = ((H2**2.1) % 11) * ((I2**2.3) % 13) + ((J2**2.5) % 17) * ((K2**2.7) % 19) + ((L2**2.9) % 23) * ((M2**3.1) % 29) + ((N2**3.3) % 31) * ((O2**3.5) % 37)
    P2,Q2,R2 = day_mod, alt_sum_prod_1, alt_sum_prod_2

    val = (P2 ** (1 + (R2 % 1)) % math.pi) * Q2
    return math.floor(1000000 * (val % 1))



if __name__ == "__main__":
    all_text = ""
    with open("dracula.txt") as f:
        all_text += f.read()
    with open("war-and-peace.txt") as f:
        all_text += f.read()
    with open("monte-cristo.txt") as f:
        all_text += f.read()

    all_text = all_text

    vals = []
    n_samples = 10000
    min_len = 6
    max_len = 150
    for i in range(n_samples):
        s = get_random_substring(all_text, min_len, max_len)
        val = get_rand_val(s)
        vals.append(val)
        print(f"\nstring:\n----vvv----\n{s}\n----^^^----\ngave value {val}")
        plot_trajectory_over_days(s, n_days=10000)

    plt.hist(vals, bins=100)
    plt.show()
