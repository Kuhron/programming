import re
import random

semitic_pronunciation = "S AH0 M IH1 T IH0 K"
semitic_segments = semitic_pronunciation.split()
regex_itic = "IH[1-9] T IH0 K$"
regex_etic = "EH[1-9] T IH0 K$"
regex_three_vowels = "^[^\d]*\d[^\d]*\d[^\d]*\d[^\d]*\d[^\d]*$"  # note that there is always a number for pronunciation designation of homographs

with open("cmudict.txt") as f:
    lines = f.readlines()
lines = [l.strip() for l in lines]
lines_itic = [l for l in lines if re.search(regex_itic, l)]
lines_etic = [l for l in lines if re.search(regex_etic, l)]

print(f"there are {len(lines_itic)} -itic words")
# print("sample:")
# for x in random.sample(lines_itic, 5):
#     print(x)
print(f"there are {len(lines_etic)} -etic words")
# print("sample:")
# for x in random.sample(lines_etic, 5):
#     print(x)

words_itic = [l.split()[0] for l in lines_itic]
words_etic = [l.split()[0] for l in lines_etic]

print("-itic: " + " ".join(words_itic))
print()
print("-etic: " + " ".join(words_etic))

lines_itic_3syll = [l for l in lines_itic if re.match(regex_three_vowels, l)]
lines_etic_3syll = [l for l in lines_etic if re.match(regex_three_vowels, l)]

print(f"there are {len(lines_itic_3syll)} -itic words with 3 syllables")
print(f"there are {len(lines_etic_3syll)} -etic words with 3 syllables")

words_itic_3syll = [l.split()[0] for l in lines_itic_3syll]
words_etic_3syll = [l.split()[0] for l in lines_etic_3syll]

print("-itic 3 syllables: " + " ".join(words_itic_3syll))
print("-etic 3 syllables: " + " ".join(words_etic_3syll))
