import re

unicode_fp = "unicode_chars.csv"
kmn_fp = "IPA14.kmn"
ustr_pattern = "U\+[0-9a-fA-F]{4}"

store_ustr_pattern = "^store\([^)]+\) " + ustr_pattern + "$"

with open(kmn_fp) as f:
    lines = f.readlines()
for l in lines:
    assert l[-1] == "\n" and l.count("\n") == 1
lines = [l.replace("\n", "") for l in lines]

store_ustr_lines = [l for l in lines if re.search(store_ustr_pattern, l)]
print("store_ustr_lines:")
for x in store_ustr_lines: print(x)
print("----\n")

ustr_lines = [l for l in lines if re.search(ustr_pattern, l) and l not in store_ustr_lines]
print("ustr_lines:")
for x in ustr_lines: print(x)
print("----\n")

with open(unicode_fp) as f:
    lines = f.readlines()
for l in lines:
    assert l[-1] == "\n" and l.count("\n") == 1
lines = [l.replace("\n", "") for l in lines]
rows = [l.split(",") for l in lines]
assert all(len(x) == 3 for x in rows)

ustr_to_name = {}
ustr_to_char = {}
name_to_ustr = {}
char_to_ustr = {}
for ustr, char, name in rows:
    assert ustr not in ustr_to_name
    assert ustr not in ustr_to_char
    assert name not in name_to_ustr
    assert char not in char_to_ustr
    
