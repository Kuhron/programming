import os
import random
from pathlib import Path
import subprocess
import time
import sys

every_n_minutes = int(sys.argv[1]) if len(sys.argv) > 1 else None
base_dir = "/home/wesley/Desktop/Learning/"
pdf_files = list(Path(base_dir).rglob("*.pdf"))

while True:
    chosen = random.choice(pdf_files)
    print("chose PDF: {}".format(chosen))
    subprocess.Popen(["xdg-open", chosen], start_new_session=True)
    if every_n_minutes is None:
        print("done")
        break
    elif every_n_minutes < 1:
        raise ValueError("rate limit")
    else:
        print(f"another pdf will open in {every_n_minutes} minutes")
        time.sleep(every_n_minutes * 60)
