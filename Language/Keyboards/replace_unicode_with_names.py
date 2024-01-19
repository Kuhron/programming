import re
import os


unicode_fp = "unicode_chars.csv"
kmn_fp = "IPA14.kmn"
ustr_pattern = "U\+[0-9a-fA-F]{4}"

store_ustr_pattern = "^store\([^)]+\) " + ustr_pattern + "$"

with open(kmn_fp) as f:
    lines = f.readlines()
for l in lines:
    assert l[-1] == "\n" and l.count("\n") == 1
kmn_lines = [l.replace("\n", "") for l in lines]

store_ustr_lines = [l for l in kmn_lines if re.search(store_ustr_pattern, l)]
print("store_ustr_lines:")
for x in store_ustr_lines: print(x)
print("----\n")

ustr_lines = [l for l in kmn_lines if re.search(ustr_pattern, l) and l not in store_ustr_lines]
print("ustr_lines:")
for x in ustr_lines: print(x)
print("----\n")

with open(unicode_fp) as f:
    lines = f.readlines()
for l in lines:
    assert l[-1] == "\n" and l.count("\n") == 1
csv_lines = [l.replace("\n", "") for l in lines]
rows = [l.split(",") for l in csv_lines]
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
    assert ustr == ustr.lower(), ustr
    ustr_to_name[ustr] = name
    ustr_to_char[ustr] = char
    name_to_ustr[name] = ustr
    char_to_ustr[char] = ustr

# now replace the Unicode point strings with the character name
new_lines = []
for l in kmn_lines:
    if l in ustr_lines:
        assert l not in store_ustr_lines
        # who cares about optimizing this, it's not a big file so I can be redundant
        ustr_matches = re.findall(ustr_pattern, l)
        print(ustr_matches)
        for ustr in ustr_matches:
            name = ustr_to_name[ustr.lower()]
            l = l.replace(ustr, "$"+name)
        new_l = l
    else:
        new_l = l
    new_lines.append(new_l)

# just print all the store lines I want to add, and manually paste them into the kmn file
for ustr, name in sorted(ustr_to_name.items()):
    print(f"store({name}) {ustr.upper()}")

output_fp = "new.kmn"
assert not os.path.exists(output_fp)
with open(output_fp, "w") as f:
    for l in new_lines:
        assert "\n" not in l
        f.write(l + "\n")

print("done")
