"""
raw = ""
entry = input("Paste text: ")
while entry != "":
    raw = raw + entry
    entry = input("")
"""

import sys
print("Input text, then press Ctrl+D (in IDLE).\n")
raw = sys.stdin.read()

finals = {}
for i in range(len(raw)-1):
    if raw[i+1] in [" ", ".", ",", "\n"]:
        if raw[i] in finals:
            finals[raw[i]] += 1
        else:
            finals[raw[i]] = 1

# how to sort a dict by value (replace the 1 with a 0 to sort by key)
import operator
finals_sorted = sorted(finals.items(), key = operator.itemgetter(1), reverse = True)

print(finals_sorted)
