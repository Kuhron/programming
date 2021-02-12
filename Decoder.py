import string
import sys

abc = string.ascii_lowercase


def shift(s, n):
    indices = [abc.index(x) if x in abc else None for x in s]
    indices = [(x+n) % 26 if x is not None else None for x in indices]
    res = ""
    for index, c in zip(indices, s):
        if index is None:
            res += c
        else:
            res += abc[index]
    return res


def print_all_shifts(s, header=True):
    if header:
        print("\n-- Caesar shifts")
    for i in range(26):
        print("{} : {}".format(i, shift(s,i)))
    print("--")


def atbash(s):
    res = ""
    for x in s:
        if x in abc:
            res += abc[25 - abc.index(x)]
        else:
            res += x
    return res


def print_atbash_shifts(s):
    print("\n-- Atbash (reversed) Caesar shifts:")
    s = atbash(s)
    print_all_shifts(s, header=False)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        s = sys.argv[1]
    else:
        print("string to try to decode:")
        s = input()

    print_all_shifts(s)
    print_atbash_shifts(s)
