# high idea: a string type that can only be operated on using arithmetic about its chars
# so to concatenate and get "a" + "b" -> "ab", you need to do match to get the string as a base-128 number (ASCII)
# mimic all string operations like add, replace, reverse, slicing, etc., using this


import math


class ArithmeticString(int):
    def __init__(self, s):
        assert type(s) is str
        self.n = str_to_int(s)

    def __repr__(self):
        return int_to_str(self.n)

    def __add__(self, other):
        if type(other) is not ArithmeticString:
            return NotImplemented
        return ArithmeticString(self.n + other.n)

    def replace(self, substr1, substr2):
        raise NotImplementedError


def str_to_num(s):
    orig_s = s
    c0 = chr(0)
    n_leading_zeros = 0
    while len(s) > 0 and s[0] == c0:
        n_leading_zeros += 1
        s = s[1:]
    frac = int_to_frac(n_leading_zeros)
    n = 0
    for i in range(len(s)):
        ith_char_from_right = s[-(i+1)]
        x = ord(ith_char_from_right)
        assert 0 <= x < 128, f"need ASCII, got {ith_char_from_right!r} of ord {x}"
        n += x * (128 ** i)
    res = n + frac
    print(f"str_to_num({orig_s!r}) = {res}")
    return res


def num_to_str(n):
    # problem with leading zeros in the strings, want strings with ord lists like [0, 0, 4, 0, 0] to be distinct from those like [4,]
    # could use decimal part to encode number of leading zeros
    # decimal of 0 means no leading zeros
    # decimal of 1/2 means 1, then 1/4 : 2, 3/4: 3, and so on through the odd numerators over powers of two
    if n == 0:
        return ""
    s = ""
    n_leading_zeros = frac_to_int(n % 1)
    if n_leading_zeros > 1074:
        print(f"assuming fractional part of {n % 1} is actually zero")
        n_leading_zeros = 0
    c0 = chr(0)
    leading_zeros = c0 * n_leading_zeros
    n = int(n)
    while n > 0:
        n, m = divmod(n, 128)
        s += chr(m)
    res = leading_zeros + s
    print(f"num_to_str({n}) = {res!r}")
    return res


def frac_to_int(x):
    # can't just invert int_to_frac because of floor function
    assert 0 <= x < 1, x
    if x == 0:
        return 0

    # find lowest p such that x is a multiple of 2**-p
    p = 0
    while True:
        if p > 1074:
            raise Exception("got too many leading zeros")
        m = 2**-p
        if m == 0:
            raise Exception("got too many leading zeros, float cannot store more than 1074 of them")
            # because 2**-1074 != 0 but 2**-1075 == 0
        if x % m == 0:
            break
        p += 1
    # print(f"{x=}, {m=}")
    d1 = x
    e1 = m
    f1 = 1/e1
    assert f1 % 1 == 0
    assert math.log(f1, 2) % 1 == 0
    g1 = d1/e1
    h1 = (g1+1)/2
    i1 = f1/2
    j1 = h1+i1-1
    n = j1
    assert n % 1 == 0
    n = int(n)
    return n


def int_to_frac(n):
    if n == 0:
        return 0
    c1 = n
    d1 = math.log(c1,2)
    e1 = math.floor(1+d1)
    f1 = 2**e1
    g1 = f1-c1
    h1 = 2*g1-1
    i1 = f1-h1
    a1 = i1
    b1 = f1
    return a1/b1



if __name__ == "__main__":
    a = ArithmeticString  # shorthand

    print(f"{int_to_frac(37) = }")
    print(f"{frac_to_int(275/1024) = }")
    print(f'{str_to_num("shorthand") = }')
    print(f"{num_to_str(5743289830139247) = }")
    c0 = chr(0)
    print(f'{str_to_num(c0*3 + "456" + c0*2) = }')
    print(f"{num_to_str(474382987 + 17/32) = }")

    # tests
    assert int_to_frac(7) == 7/8
    assert int_to_frac(12) == 9/16
    assert frac_to_int(7/8) == 7
    assert frac_to_int(9/16) == 12
    fn = lambda n: str_to_num(num_to_str(n)) == n
    fs = lambda s: num_to_str(str_to_num(s)) == s
    assert fn(0)
    assert fn(89/128)
    assert fn(2344838883)
    assert fn(3118 + 555/2048)
    assert fs(c0*7 + "1918")
    assert fs("asbgf" + c0*3)
    assert fs("".join(chr(random.randint(0, 127)) for i in range(17)))
    assert a("aaa").replace(a("aa"),a("bb")) == a("bba")
    assert a("bc") + a("ef") == a("bcef")
    assert a("abracadabra").replace(a("br"), a("tz")) == a("atzacadatza")

