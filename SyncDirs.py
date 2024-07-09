# script to compare the hashed contents of files in two dirs
# and see which files are the same
# user will be prompted for where to put each duplicate (one of the two relative paths)
# anything that doesn't have a counterpart in the other dir will be left where it is

import os
import shutil
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


def compare_hash_dicts(d1, d2, dir1, dir2, out_dir):
    ks1 = set(d1.keys())
    ks2 = set(d2.keys())
    ks = ks1 | ks2
    ks = sorted(ks)

    copy_operations_to_make = []

    print(f"\n{len(ks)} unique files found\n")
    for hx in ks:
        print(hx)

        in_d1 = hx in d1
        in_d2 = hx in d2
        assert in_d1 or in_d2, "key shouldn't exist"
        fps1 = d1.get(hx, [])
        fps2 = d2.get(hx, [])
        fps1 = sorted(fps1)
        fps2 = sorted(fps2)

        print_files_in_dir(dir1, fps1)
        print_files_in_dir(dir2, fps2)

        # cases

        # if it is in only one dir then it should have that relative path in the new dir
        if in_d1 and (not in_d2):
            src_fp = os.path.join(dir1, fps1[0])  # just pick any of them
            dest_fp = os.path.join(out_dir, get_fp_choice(fps1))
        elif (not in_d1) and in_d2:
            src_fp = os.path.join(dir2, fps2[0])
            dest_fp = os.path.join(out_dir, get_fp_choice(fps2))

        # if it is in both dirs then user needs to pick where it will go
        elif in_d1 and in_d2:
            fps_combined = sorted(set(fps1) | set(fps2))
            src_fp = os.path.join(dir1, fps1[0])
            dest_fp = os.path.join(out_dir, get_fp_choice(fps_combined))
        else:
            raise Exception("unhandled case")

        pair = (src_fp, dest_fp)
        print(f"\nnew copy operation:\n    {src_fp}\n--> {dest_fp}")

        copy_operations_to_make.append(pair)
        print()

    return copy_operations_to_make


def compare_dir_contents(dir1, dir2, out_dir):
    d1 = get_hashes_to_relative_paths(dir1)
    d2 = get_hashes_to_relative_paths(dir2)
    copy_operations = compare_hash_dicts(d1, d2, dir1, dir2, out_dir)
    return copy_operations


def print_files_in_dir(dr, fps):
    if len(fps) == 0:
        print(f"<<< absent  in {dr!r} >>>")
    else:
        print(f"<<< present in {dr!r} >>>")
        for fp in fps:
            print(f"- {fp}")


def get_fp_choice(fps):
    if len(fps) == 1:
        return fps[0]

    print("\nPlease choose which of the following filepaths should be used for this file in the synced directory:")
    for i, fp in enumerate(sorted(fps)):
        print(f"- {i+1}. {fp!r}")
    print()
    while True:
        n = input("Please enter the number of the desired filepath: ").strip()
        try:
            n = int(n)
        except ValueError:
            print("invalid input")
            continue
        if not (0 <= n-1 < len(fps)):
            print("invalid input")
            continue
        i = n - 1
        choice = fps[i]
        break
    return choice


def verify_all_files_were_copied(dir1, dir2, out_dir):
    d1 = get_hashes_to_relative_paths(dir1)
    d2 = get_hashes_to_relative_paths(dir2)
    out_dir = get_hashes_to_relative_paths(out_dir)
    missing = []
    for k in set(d1.keys()) | set(d2.keys()):
        if k not in out_dir.keys():
            missing.append(k)
    if len(missing) > 0:
        raise Exception("failure to copy all files! FIXME there is a bug somewhere")
    else:
        print("all files in either of the source directories were successfully copied to the destination directory")


if __name__ == "__main__":
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    out_dir = sys.argv[3]
    assert not os.path.exists(out_dir)
    os.mkdir(out_dir)

    copy_operations = compare_dir_contents(dir1, dir2, out_dir)

    for src_fp, dest_fp in copy_operations:
        os.makedirs(os.path.dirname(dest_fp), exist_ok=True)
        shutil.copy(src_fp, dest_fp)

    verify_all_files_were_copied(dir1, dir2, out_dir)
