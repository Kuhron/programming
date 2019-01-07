# JUMA 2

d = {
    "A" : "o",
    "B" : "v",
    "C" : "x",
    "D" : "z",
    "E" : "a",
    "F" : "tʃ",
    "G" : "d",
    "H" : "k",
    "I" : "e",
    "J" : "ɣ",
    "K" : "t",
    "L" : "n",
    "M" : "w",
    "N" : "r",
    "O" : "u",
    "P" : "f",
    "Q" : "g",
    "R" : "j",
    "S" : "p",
    "T" : "s",
    "U" : "i",
    "V" : "dʒ",
    "W" : "m",
    "X" : "ŋ",
    "Y" : "l",
    "Z" : "b",
}

def f(s):
    print(s.upper())
    return "".join(d.get(c, c) for c in s.upper())


if __name__ == "__main__":
    while True:
        s = input("string to translate: ")
        print(f(s))
