d = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety"
}

def s(n):
    if n == 1000: return "onethousand"
    assert 1 <= n <= 1000
    res = ""
    h = n // 100
    n -= h * 100
    t = n // 10
    n -= t * 10
    o = n
    if h > 0:
        res += d[h] + "hundred" + ("and" if t > 0 or o > 0 else "")
    if t == 1:
        res += d[10*t + o]
    else:
        res += (d[10*t] if t > 0 else "") + (d[o] if o > 0 else "")
    return res

ls = lambda n: len(s(n))

# debug
for n in range(1, 212):
    print(n, s(n), ls(n))


print(sum(ls(n) for n in range(1, 1001)))
