import re

fp = "/home/wesley/Desktop/tpayne-pragm-test.tex"

with open(fp) as f:
    lines = f.readlines()

indices_to_edit = set()
for i, line in enumerate(lines):
    if r"\gll" in line:
        indices_to_edit.add(i)
        indices_to_edit.add(i+1)  # next line in addition to this one
        print(f"will edit lines {i} and {i+1}")

pattern = "-"
repl = r" \\textendash{} "

new_lines = []
for i, line in enumerate(lines):
    if i in indices_to_edit:
        new_line = re.sub(pattern, repl, line)
        print(f"\nline {i} changed from\n{line}\nto\n{new_line}\n---")
    else:
        new_line = line
    new_lines.append(new_line)

with open(fp.replace(".tex", "-replaced.tex"), "w") as f:
    f.writelines(new_lines)

