# messing around with colexifications using CLICS database
# citation: Rzymski, Tresoldi et al. 2019. The Database of Cross-Linguistic Colexifications, reproducible analysis of cross- linguistic polysemies. DOI: doi.org/10.17613/5awv-6w15

# instructions for downloading/compiling the data: https://github.com/clics/clics3
# I have already done this on 2021-07-09 but am not committing all that stuff to my repo (it's in other repos out there as listed in datasets.txt)
# to check what datasets you have, run `clics datasets` in a terminal

# want to be able to mess with CLICS data my own way, not just run their code to create the GML graph
# e.g. take some subset of languages and make a semantic map out of just those ones
# e.g. perform random walk on concepts to create new semantic map for conlang


import os
import csv
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_forms_filepaths():
    directory = "/home/wesley/programming/Language/src/"
    subdirs = [f for f in os.scandir(directory) if f.is_dir()]
    filename = "forms.csv"
    fps = []
    for subdir in subdirs:
        if subdir.name == "lexibank-hantganbangime":
            # for some reason this one has an extra src/ and a subdir with the same name EXCEPT hyphen is replace with underscore. WHY??
            file_dir = "src/lexibank_hantganbangime/cldf"
        else:
            file_dir = "cldf"

        fp = os.path.join(directory, subdir, file_dir, filename)
        if os.path.exists(fp):
            fps.append(fp)
        else:
            print(f"Warning: forms file does not exist: {fp}")
    return fps


def get_rows_from_fp(fp):
    with open(fp) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    for row in rows:
        assert "fp" not in row
        row["fp"] = fp  # store which file this record came from
    return rows


def get_rows_from_fps(fps):
    rows = []
    all_keys = set()
    keys_in_all_fps = None
    for fp in fps:
        fp_rows = get_rows_from_fp(fp)
        fp_keys = set(fp_rows[0].keys())
        all_keys |= fp_keys
        if keys_in_all_fps is None:
            keys_in_all_fps = fp_keys
        else:
            keys_in_all_fps &= fp_keys
            # don't want to do this with initializing it as empty set because then it will just stay empty
        rows += fp_rows

    # some of them don't have the same keys, leave those keys out (don't set them to some default)
    # so that KeyError is raised if you try to use one
    print(f"all keys:\n{sorted(all_keys)}\nkeys in all files:\n{sorted(keys_in_all_fps)}")
    return rows, all_keys, keys_in_all_fps


def print_keys_of_rows(rows, keys):
    max_key_len = max(len(k) for k in keys)
    keys = sorted(keys)
    for i, row in enumerate(rows):
        print(f"row {i}")
        for k in keys:
            val = row.get(k)
            print(f"- {k.ljust(max_key_len+1)}: {val}")
        print()


def show_key_statistics(rows, keys):
    # for each key, show how many rows have it (and proportion) and some examples of what is in it
    for k in sorted(keys):
        rows_with_key = [row for row in rows if k in row and row[k] != ""]
        print(f"key {k} is in {len(rows_with_key)} rows out of {len(rows)} ({100*len(rows_with_key)/len(rows):.2f}%). Examples of its values:")
        sample_rows = random.sample(rows_with_key, min(5, len(rows_with_key)))
        for row in sample_rows:
            print(f"{k} : {row[k]}")
        print()


if __name__ == "__main__":
    fps = get_forms_filepaths()
    rows, all_keys, keys_in_all_fps = get_rows_from_fps(fps)

    # sample_rows = random.sample(rows, 10)
    # print_keys_of_rows(sample_rows, all_keys)
    # print_keys_of_rows(sample_rows, keys_in_all_fps)
    # show_key_statistics(rows, all_keys)

    # Form is in all rows, which is good, but can't find a concept-like thing in every row
    # Parameter_ID is in 91% of them, and it references a file parameters.csv that tells which Concepticon concept is referred to
    rows_without_parameter_id = [row for row in rows if "Parameter_ID" not in row]
    fps_without_parameter_id = set(row["fp"] for row in rows_without_parameter_id)
    print(fps_without_parameter_id)  # it's just pylexirumah that does this
    # pylexirumah uses Concept_ID instead, referencing concepts.csv, which has Concepticon_ID connected to the concept name
    # and all other databases have Concepticon_ID in their parameters.csv (I think), as well as Concepticon_Gloss


