# script to compare the hashed contents of files in two dirs
# and see which files are the same
# user will be prompted for where to put each duplicate (one of the two relative paths)
# anything that doesn't have a counterpart in the other dir will be left where it is

import os
import sys
from hashlib import sha256


def h(fp):
    with open(fp, "rb") as f:
        hx = sha256(f.read()).hexdigest()
    return hx


def get_hashes_to_relative_paths(parent_dir):
    d = {}
    for dirpath, subdirs, filenames in os.walk(parent_dir):
        for fn in filenames:
            abs_fp = os.path.join(dirpath, fn)
            rel_fp = os.path.relpath(abs_fp, parent_dir)
            hx = h(abs_fp)
            if hx not in d:
                d[hx] = []
            d[hx].append(rel_fp)
    return d


def check_hash_dict_contents(d):
    for hx, fps in d.items():
        print(hx)
        for fp in fps:
            print(f"- {fp}")
        print()
        if len(fps) > 1:
            input("a\n")


def compare_hash_dicts(d1, d2, dir1, dir2):
    ks1 = set(d1.keys())
    ks2 = set(d2.keys())
    ks = ks1 | ks2
    ks = sorted(ks)

    for hx in ks:
        print(hx)

        in_d1 = hx in d1
        in_d2 = hx in d2
        fps1 = d1.get(hx, [])
        fps2 = d2.get(hx, [])

        if in_d1 and in_d2:
            
        elif (not in_d1) and in_d2:
            print(f"<<< absent in {dir1!r} >>>")
            ?
        elif in_d1 and (not in_d2):
            print(f"<<< absent in {dir2!r} >>>")
        else:
            raise Exception("key shouldn't exist")

        print()


def compare_dir_contents(dir1, dir2):
    d1 = get_hashes_to_relative_paths(dir1)
    d2 = get_hashes_to_relative_paths(dir2)
    compare_hash_dicts(d1, d2, dir1, dir2)



if __name__ == "__main__":
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    compare_dir_contents(dir1, dir2)

