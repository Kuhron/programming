#repl.it is still in 2.x, update when using at home
#updated for 3.x

import string

wordlist = []
raw = input("Enter or paste the list of words, separated by spaces. Non-alphabetic characters will be ignored.\n").lower()

working = ""
for i in raw:
    if i != " " and i != "\n": #note: the \n thing here does not actually work. input still must include spaces and not just line breaks
        working = working + i
    else:
        wordlist.append(working)
        working = ""
wordlist.append(working)

cleanlist = []

#this is inefficient, so use maketrans to remove non-letters if you can figure out how
for u in wordlist:
    lower = ""
    for i in u:
        if i in string.ascii_lowercase:
            lower = lower + i
    cleanlist.append(lower)

armenglish_flip = str.maketrans("abcdefghijklmnopqrstuvwxyz","aecpbuhgtyqosrldknmifzwxjv")

print("\n")

flippedlist = []
for u in cleanlist:
    v = (u.translate(armenglish_flip))[::-1]
    flippedlist.append(v)
    if v in cleanlist:
        print("\"%s\" upside-down is \"%s\"" % (u, v))
        cleanlist.remove(v)
        
print("\nHere are the corresponding flipped words. Note that the words themselves are still in forward order.\n")
print(" ".join(flippedlist))
