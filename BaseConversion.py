from string import ascii_uppercase as up
from string import ascii_lowercase as lo

def to_base(x, b): #b is target base, x is in base 10
    if b < 2 or b > 62 or b % 1 != 0:
        raise ValueError("Invalid base {0}".format(b))
    else:
        result = ""
        left = x
        i = 0
        while left > 0:
            d = (left % (b ** (i+1)))/(b ** i)
            if d > 9 and d <= 35:
                result = up[int("%d" % (d - 10))] + result
            elif d > 35:
                result = lo[int("%d" % (d - 36))] + result
            else:
                result = ("%d" % d) + result
            left = left - (d * (b ** i))
            i += 1
        return result if result != "" else "0"

def from_base(x, b): #x is in base b, will be converted to base 10
    s = str(x)
    result = 0
    for i in range(len(s)):
        d = s[len(s) - i - 1]
        if d in up:
            add = (up.index(d) + 10) * (b ** i)
        elif d in lo:
            add = (lo.index(d) + 36) * (b ** i)
        else:
            add = int(d) * (b ** i)
        result = result + add
    return result
#this will actually allow using digits that are too high for a base, because that's fun and I don't care to restrict it

def from_base_to_base(x, bi, bf):
    return to_base(from_base(x, bi), bf)


if __name__ == "__main__":
    for j in range(2,21):
        print(from_base(100, j))
    print("--")
    for j in range(2,21):
        print(from_base_to_base("4Ec7k0R", 62, j))
