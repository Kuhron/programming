# see amazing tutorial at https://www.instructables.com/Python-coding-for-Minecraft/


import os
from shutil import copyfile

source_dir = "/home/wesley/programming/Minecraft/"
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
        inp = input("overwrite? [y/n]")
        if inp == "y":
            print("overwriting")
        else:
            print("skipping")
            continue  # important!
    copyfile(source_fp, target_fp)
    print("copied {} -> {}".format(source_fp, target_fp))

print("done")
