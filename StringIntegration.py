def add_strs(a, b):
    a = list(a.strip())
    b = list(b.strip())
    if len(b) > len(a):
        a, b = b, a
    n = len(a)
    diff = len(a) - len(b)
    front_blanks = diff // 2
    b = [""] * front_blanks + b + [""] * (diff - front_blanks)
    res = ""
    for i in range(n):
        res += a[i] + b[i]
    return res


def f(s):
    s = s.split()
    res = ""
    for x in s:
        res = add_strs(res, x)
    return res


if __name__ == "__main__":
    x = input("input: ")
    print(f(x))
