# see amazing tutorial at https://www.instructables.com/Python-coding-for-Minecraft/


import os
import sys
from shutil import copyfile

source_dir = "/home/wesley/programming/Minecraft/RaspberryJamScripts/"
target_dir = "/home/wesley/.minecraft/mcpipy/"

pys_in_source_dir = [x for x in os.listdir(source_dir) if x.endswith(".py")]

for f in pys_in_source_dir:
    source_fp = os.path.join(source_dir, f)
    target_fp = os.path.join(target_dir, f)
    if f == __file__:
        print("skipping {} which is this script".format(f))
        continue  # important!
    if os.path.exists(target_fp):
        print("file exists: {}".format(target_fp))
        force_overwrite = len(sys.argv) > 1 and sys.argv[1] == "-f"
        if not force_overwrite:
            inp = input("overwrite? [y/n]")
        else:
            inp = None

        if force_overwrite or inp == "y":
            print("overwriting {}".format(f))
        else:
            print("skipping {}".format(f))
            continue  # important!
    copyfile(source_fp, target_fp)
    print("copied {} -> {}".format(source_fp, target_fp))

print("done")
