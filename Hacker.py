import random
import os
from pathlib import Path

print("loading...")

directory = "/home/kuhron/programming/"

py_files = list(Path(directory).rglob('*.py'))

py_lines = []
for fp in py_files:
    print(f"loading {fp}")
    with open(fp) as f:
        lines = f.readlines()
    for line in lines:
        py_lines.append(line.replace("\n", ""))

while True:
    line = random.choice(py_lines)
    print(line)
