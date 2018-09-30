# random word from Douglas Adams' "The Meaning Of Liff"

import random

with open("liff.txt") as f:
    lines = f.readlines()

def is_entry(line):
    s = line.strip().replace(" ", "").split("(")[0]
    return s != "" and s.upper() == s

all_strs = []
current_str = ""
for line in lines:
    if is_entry(line):
        if current_str != "":
            all_strs.append(current_str)
        current_str = line
    elif line.strip() != "":
        current_str += line

print(random.choice(all_strs))
