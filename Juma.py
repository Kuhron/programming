# JUMA 2

d = {
    "a" : "o",
    "b" : "v",
    "c" : "ts",
    "d" : "z",
    "e" : "a",
    "f" : "tʃ",
    "g" : "d",
    "h" : "k",
    "i" : "e",
    "j" : "dz",
    "k" : "t",
    "l" : "n",
    "m" : "w",
    "n" : "r",
    "o" : "u",
    "p" : "f",
    "q" : "tɕ",
    "r" : "y",  # /j/
    "s" : "p",
    "t" : "s",
    "u" : "i",
    "v" : "dʒ",
    "w" : "m",
    "x" : "ɕ",
    "y" : "l",
    "z" : "b",
}

def f(s):
    return "".join(d.get(c, c) for c in s.lower())


if __name__ == "__main__":
    while True:
        s = input("string to translate: ")
        print(f(s))
