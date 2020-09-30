import os
import random
from pathlib import Path
import subprocess


base_dir = "/home/wesley/Desktop/Learning/"
pdf_files = list(Path(base_dir).rglob("*.pdf"))
chosen = random.choice(pdf_files)
print("chose PDF: {}".format(chosen))
subprocess.call(["xdg-open", chosen])

