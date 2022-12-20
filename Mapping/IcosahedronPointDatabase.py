# store data according to point code
# all of them need the same header
# but need to allow sparseness, e.g. many points won't have values for certain variables, e.g. suppose "salt flat" is undefined most places but has values in small regions of the world, don't want to waste space having that defined as zero for almost every point
# maybe have a format that indexes the variables, every time a new variable is added it gets the next index number
# there is a global file telling which index means which variable, e.g. {0:is_land, 1:elevation, 2:volcanism, etc.}
# then the rows in the files look something like: 40128,0=0,1=-65.3,5=17.1,24=0
# so they're only showing the variables they're specified for


import os
import pathlib
import re
import sys
import math
import random
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import IcosahedronMath as icm
from BiDict import BiDict
from PointCodeTrie import PointCodeTrie
import FindPointsInCircle as find
import PlottingUtil as pu
from LoadMapData import get_default_values_of_conditions, translate_array_by_dict


class IcosahedronPointDatabase:
    DTYPE = np.dtype("int64")  # all values must always be int
    VAL_TYPES = [DTYPE, int]
    MAX_INT_VALUE = 2**63-1
    SPECIAL_COLUMN_NAMES = ["index", "prefix_lookup_number"]

    def __init__(self):
        pass

    @staticmethod
    def new(root_dir, n_point_code_chars_per_level):
        db = IcosahedronPointDatabase()
        db.root_dir = root_dir
        if os.path.exists(db.root_dir):
            dir_is_empty = len(os.listdir(db.root_dir)) == 0
            if not dir_is_empty:
                raise Exception(f"root_dir not empty: {db.root_dir}")
        else:
            os.mkdir(db.root_dir)

        with open(os.path.join(root_dir, "IcosahedronPointDatabase.txt"), "w") as f:
            f.write("This is an IcosahedronPointDatabase.")
        db.metadata_file = os.path.join(root_dir, "metadata.txt")

        db.variables_dict = BiDict(int, str)
        db.metadata = {
            "n_point_code_chars_per_level": n_point_code_chars_per_level,
        }
        db.write_metadata()
        return db

    @staticmethod
    def load(root_dir):
        print(f"loading database from {root_dir}")
        db = IcosahedronPointDatabase()
        db.root_dir = root_dir
        db.metadata_file = os.path.join(root_dir, "metadata.txt")
        db.data_file = os.path.join(root_dir, "data.h5")
        db.metadata = IcosahedronPointDatabase.get_metadata_from_file(db.metadata_file)
        db.read_hdf()
        print(f"done loading database from {root_dir}")
        return db

    @staticmethod
    def db_exists(root_dir):
        if not os.path.exists(root_dir):
            return False
        contents = os.listdir(root_dir)
        necessary_contents = ["IcosahedronPointDatabase.txt", "variables.txt", "metadata.txt", "data.h5"]  # dirs don't end with slash in the listing
        found = [x in contents for x in necessary_contents]
        if all(found):
            return True
        elif not any(found):
            return False
        else:
            raise Exception(f"database partially exists: {root_dir}")

    def add_variable(self, variable_name):
        assert variable_name not in self.df.columns, f"variable {variable_name} already in database"
        self.df.insert(len(self.df.columns), variable_name, value=np.nan)

    def write_metadata(self):
        d = self.metadata
        assert type(d) is dict
        lines = []
        for k,v in sorted(d.items()):
            l = f"{k}:{v}"
            lines.append(l)
        s = "\n".join(lines)
        with open(self.metadata_file, "w") as f:
            f.write(s)

    def write_hdf(self):
        self.validate()
        self.df.to_hdf(self.data_file, key="data")
        print("IcosahedronPointDatabase written to file.\n")

    def read_hdf(self):
        self.df = pd.read_hdf(self.data_file)
        self.validate()

    def validate(self):
        self.verify_df_dtype()
        self.check_no_trailing_zeros_in_point_codes()
        self.check_no_duplicate_point_codes()
        self.check_no_duplicate_prefix_lookups()

    def verify_df_dtype(self):
        verify_df_dtype(self.df)
    
    def check_no_trailing_zeros_in_point_codes(self):
        check_no_trailing_zeros_in_point_codes(self.df.index)
    
    def check_no_duplicate_point_codes(self):
        assert self.df.index.has_duplicates is False
        # don't do `not object.attr` because that will be falsy if the attribute is absent
    
    def check_no_duplicate_prefix_lookups(self):
        self.fill_lookup_numbers()
        lns = self.df["prefix_lookup_number"]
        duplicated_mask = lns.duplicated(keep=False)  # for some reason, keep=False means all duplicates are marked as True in the returned bool series
        if duplicated_mask.sum() > 0:
            nums_with_conflicts = np.unique(lns[duplicated_mask])
            for ln in nums_with_conflicts:
                this_ln_mask = lns == ln
                pcs_this_ln = self.df.index[this_ln_mask]
                print(f"point codes with lookup number {ln}: {pcs_this_ln}")
            raise RuntimeError("duplicate lookup numbers created; see above")

    def get_variables(self):
        return sorted(self.df.columns)

    def get_n_variables(self):
        return len(self.get_variables())
    
    def get_condition_variables(self):
        return sorted(x for x in self.df.columns if x.endswith("_condition"))
    
    def get_value_variables(self):
        return sorted(x for x in self.df.columns if not x.endswith("_condition") and x not in IcosahedronPointDatabase.SPECIAL_COLUMN_NAMES)

    def get_variable_encoding_types(self):
        # based on the values in the df
        # e.g. uint3 (unsigned int with 3 bits), sint4 (signed int with 4 bits other than the sign)
        res = []
        for var_i in range(self.get_n_variables()):
            col = self.df.loc[:, var_i]
            col = col[~pd.isna(col)]  # don't count NaN in min/max
            assert (col % 1 == 0).all(), "values should be ints (even though we need to store as floats since we can't write pd.NA to hdf5"
            min_val = col.min()
            max_val = col.max()
            print(f"variable {var_i} has min val {min_val}, max val {max_val}")
            bits = math.ceil(max(math.log2(abs(min_val)+1), math.log2(abs(max_val)+1)))  # note we need n+1 bits to store 2**n, e.g. 1000 = 8
            signed = min_val < 0
            t = ("s" if signed else "u") + "int" + str(bits)
            res.append(t)
        return res

    @staticmethod
    def get_n_bits_for_encoding_type(t):
        s = t[0]
        assert s in ["s", "u"]
        signed = s == "s"
        assert t[1:4] == "int"
        n = int(t[4:])
        return n + int(signed)

    @staticmethod
    def get_metadata_from_file(fp):
        with open(fp) as f:
            lines = f.readlines()
        d = {}
        for l in lines:
            var, val = l.strip().split(":")
            d[var] = int(val)
        return d

    def get_variables_at_points(self, pcs, variable_names):
        return self.df.loc[pcs, variable_names]

    @staticmethod
    def get_all_variables_from_line(l):
        # the first index (0) in l is the point number
        d_this_line = {}
        for x in l[1:]:
            k,v = x.split("=")
            k = int(k)
            v = int(v)
            d_this_line[k] = v
        return d_this_line

    @staticmethod
    def get_variable_from_line(l, varname):
        d = IcosahedronPointDatabase.get_all_variables_from_line(l)
        val = d.get(varname)
        # typ = self.get_variable_type_from_name(variable_name)
        # for now, make everything in the db an int, can capture enums, bools, and floats to some precision, and that way I don't have to parse a file to figure out what the types are supposed to be; put units in the varname if you care about that, e.g. elevation_meters
        return val

    def set_single_point(self, pc, variable_name, value):
        check_int(value)
        self.df.loc[pc, variable_name] = value
        # self.validate()

    def set_multiple_points(self, pcs, variable_name, values):
        for val in values:
            check_int(val)
        self.df.loc[pcs, variable_name] = values
        # self.validate()

    def __getitem__(self, tup):
        pcs, varnames = tup
        
        if type(pcs) is str:
            pc = pcs
            pcs = [pc]
        if type(varnames) is str:
            varname = varnames
            varnames = [varname]
        return self.get_variables_at_points(pcs, varnames)

    def __setitem__(self, tup, val):
        pcs, variable_name = tup
        if type(pcs) is str:
            pc = pcs
            self.set_single_point(pc, variable_name, val)
        else:
            self.set_multiple_points(pcs, variable_name, val)

    def get_dict(self, pcs, varname):
        # return a dict of pc:value for just this one variable
        column = self[pcs, varname].to_dict()
        return column[varname]
    
    def get_series(self, pcs, varname):
        # return a pandas Series object for just this one variable
        return pd.Series(self.get_dict(pcs, varname))
    
    def get_single_value(self, pc, varname):
        sub_df = self[pc, varname]
        assert sub_df.size == 1, f"got more than one value at [{pc}, {varname}]"
        return sub_df.values[0,0]

    def add_value(self, pcs, varname, val):
        self.df.loc[pcs, varname] += val

    def verify_points_exist(self, pcs):
        missing = []
        for pc in pcs:
            if pc not in self.df.index:
                missing.append(pc)
        if len(missing) > 0:
            print("these points are missing from the database:", missing)
            raise KeyError("missing points; see list above")

    def write_as_images(self):
        # hacky experiment: write data in pixel RGB values in PNG files for certain iterations
        # e.g. elevation_condition has some small number of values that takes e.g. 3 bits
        # and RGB is 3*8 bits
        # so the first 3 bits of this is the elevation condition value at the point
        # we'll need to know what the possible values of the variables are
        # I know they're all ints, some variables can be signed
        # so you could have a variable with data signed int4, taking 5 bits
        # (first one for sign, rest big-endian)
        # so we have a bunch of variables and know how many bits they each take
        # and we can make a scheme to fit them together into blocks of 24
        # e.g. one image encodes variables 0, 2, 3, 4, and 8 in the bits like 00222223 33333444 44488888

        variable_encoding_types = self.get_variable_encoding_types()  # indexed in list by variable index
        # print(variable_encoding_types)
        bits_by_variable_index = [IcosahedronPointDatabase.get_n_bits_for_encoding_type(t) for t in variable_encoding_types]
        # print(bits_by_variable_index)
        
        # now partition these bit numbers into groups of 24 (greedy algorithm)
        partitions = partition_into_max_sum_groups(bits_by_variable_index, max_sum=24)
        # TODO assign point codes to pixels in images
        # TODO make images at certain resolution
        # TODO store the poles as their own image, just two pixels
        # TODO figure out how big the images should be (probably not one huge one at high iterations)
        #      e.g. divide it into watersheds
        resolution_iterations = 4
        # now make the bit arrays for each image we will write
        for variable_indices in partitions:
            arr
        im = Image.fromarray(arr)

    def get_mask_point_codes_with_prefix(self, prefix):
        # want to return true for things like "D" starting with "D00"

        # if pandas_method:  # old, this is a performance bottleneck because of the string methods
        #     return self.df.index.str.ljust(len(prefix), "0").str.startswith(prefix)
        # else:
        return self.get_mask_point_codes_with_prefixes([prefix])

    def get_all_point_codes_with_prefix(self, prefix):
        mask = self.get_mask_point_codes_with_prefix(prefix)
        return self.df.index[mask]

    def get_mask_point_codes_with_prefixes(self, prefixes):
        print("creating point code prefix mask")
        mask = np.array([False] * len(self.df.index), dtype=bool)
        if len(prefixes) == 0:
            # treat no prefixes as "nothing starts with this", empty union here is all-false
            return mask

        lookup_numbers = np.array(self.df["prefix_lookup_number"])  # pd.Series is WAY slower somehow
        for i, prefix in enumerate(prefixes):
            this_mask = get_mask_point_codes_starting_with_prefix_using_lookup_number(lookup_numbers, prefix)
            mask = mask | this_mask  # this line takes forever for pd.Series but fast for np.array
        print("done creating point code prefix mask")
        return mask

        # old 
        # mask = pd.Series([False] * len(self.df.index), index=self.df.index)
        # # without setting the mask's index to be the same as the df's it will error

        # # times for using pandas method vs manually are very similar (e.g. 86 s and 87 s)
        # if pandas_method:
        #     for prefix in prefixes:
        #         mask |= self.get_mask_point_codes_with_prefix(prefix, pandas_method=True)
        # else:
        #     max_prefix_len = max(len(x) for x in prefixes)
        #     for pc in self.df.index:
        #         pc = pc.ljust(max_prefix_len, "0")
        #         for prefix in prefixes:
        #             if pc.startswith(prefix):
        #                 mask[pc] = True
        #                 break

        print(f"done creating point code prefix mask, has {mask.sum()} items")
        return mask

    def fill_lookup_numbers(self):
        print("filling out prefix lookup numbers for point codes in database")
        if "prefix_lookup_number" not in self.df.columns:
            # don't want to deal with multi-index, just put it right after the index
            self.df.insert(0, "prefix_lookup_number", -1)
            self.df["prefix_lookup_number"] = self.df["prefix_lookup_number"].astype("int64")
        needs_number_mask = self.df["prefix_lookup_number"] == -1
        # print(self.df)
        n_needs = needs_number_mask.sum()
        if n_needs == 0:
            print("all points already have lookup numbers; done")
            return
        print(f"{n_needs} points need lookup numbers")
        lookup_numbers = get_prefix_lookup_numbers_from_point_codes(self.df.index[needs_number_mask])
        assert len(np.unique(lookup_numbers)) == len(lookup_numbers), "conflict in lookup numbers!"
        assert len(set(lookup_numbers) & set(self.df["prefix_lookup_number"])) == 0, "conflict with existing lookup numbers!"
        self.df.loc[needs_number_mask, "prefix_lookup_number"] = lookup_numbers
        # print(self.df)
        # input("df should have lookup numbers now; check")
        self.write_hdf()
        print("done filling out prefix lookup numbers for point codes in database")

    @staticmethod
    def get_line_from_dict(line_label, d):
        items = [line_label]
        for k,v in sorted(d.items()):
            items.append(f"{k}={v}")
        return ",".join(items)
    
    def write_old_block_format_to_hdf5(self):
        raise Exception("should not have to use again")
        pns = list(db.get_all_point_numbers_with_data())

        if os.path.exists("data.h5"):
            df = pd.read_hdf("data.h5")
            n_points_in_file = (~df.index.duplicated()).sum()
        else:
            n_points_in_file = 0
        
        n_pns = len(pns)
        print(f"{n_points_in_file=} / {n_pns=}")

        df = pd.DataFrame(columns=variable_indices, dtype=int)  # blank df for adding points quicker
        for i, pn in enumerate(pns):
            # if df in file has 1001 rows, last i written was 1000, 
            # so next i needs to be 1001, so skip i < n
            if i < n_points_in_file:
                continue
            if i % 100 == 0:
                print(f"{i}/{n_pns} points done")
                print(df)
            pc = icm.get_point_code_from_point_number(pn)
            variables = db.get_single_point_all_variables(pn)
            row_index = pc
            df.loc[row_index, variables.keys()] = variables.values()

            if i % 10000 == 0 or i >= n_pns - 1:
                df_in_file = pd.read_hdf("data.h5")
                df = pd.concat([df_in_file, df])
                df = df[~df.index.duplicated()]
                n_points_in_file = len(df.index)
                df.to_hdf("data.h5", key="df")
                print("wrote h5")
                print("total rows so far:", n_points_in_file)
                df = pd.DataFrame(columns=variable_indices, dtype=int)  # start clean one for next round
        
        df_in_file = pd.read_hdf("data.h5")
        assert len(df_in_file.index) == n_pns
        print("process complete; all data is stored in data.h5")


def touch(fp):
    assert not os.path.exists(fp), "cannot touch existing file"
    open(fp, "w").close()


def check_int(value):
    if type(value) not in IcosahedronPointDatabase.VAL_TYPES:
        raise TypeError(f"Database only accepts {IcosahedronPointDatabase.VAL_TYPES} values (you gave {value} of type {type(value)}).\nIf you want enum, make them int shorthands in the condition_array_dir.\nIf you want floats, choose the precision you want and make ints of that, e.g. elevation in millimeters.")


def verify_df_dtype(df):
    any_error = False
    for colname, dtype in zip(df.columns, df.dtypes):
        if dtype is not IcosahedronPointDatabase.DTYPE:
            print(f"dtype of column {colname} is {dtype}, should be {IcosahedronPointDatabase.DTYPE}")
            any_error = True
    if any_error:
        raise TypeError("df dtypes wrong; see above")


def get_point_codes_in_database_in_region(db, center_pc, d_gc, use_narrowing=True, pcs_to_consider=None):
    df = db.df
    if use_narrowing:
        assert pcs_to_consider is None, "not implemented"
        # use narrowing to get which watersheds are all inside, all outside, and split
        t0 = time.time()
        narrowing_iterations = 4 #min(resolution_iterations - 1, random.randint(0, 3))
        inside, outside, split = icm.narrow_watersheds_by_distance(center_pc, d_gc, narrowing_iterations)
        print("inside", inside)
        print("outside", outside)
        print("split", split)

        # now get point codes that are in each of these
        # label the inside ones as True, the outside ones as False
        # then filter rows by prefix to get the split ones and check them one by one
        # (see how many there are to check as well, how long this will take)
        
        inside_mask = db.get_mask_point_codes_with_prefixes(inside)
        outside_mask = db.get_mask_point_codes_with_prefixes(outside)
        split_mask = (~inside_mask) & (~outside_mask)
        split_pcs = df.index[split_mask]
        split_pcs_in_region = find.filter_point_codes_in_region_one_by_one(split_pcs, center_pc, d_gc)
        split_pc_in_region_mask = np.array([pc in split_pcs_in_region if split_mask_this_pc else False for pc, split_mask_this_pc in zip(df.index, split_mask)])
        point_in_region_mask = inside_mask | split_pc_in_region_mask
        # for pc in split_pcs:
        #     if pc in split_pcs_in_region:
        #         point_in_region_mask.loc[pc] = True
            # else:
            #     # yes it CAN happen and it's okay; raise Exception("shouldn't happen if it was already filtered?")
        print(f"there are {point_in_region_mask.sum()} points in the region")
        pcs_in_region = df.loc[point_in_region_mask].index
        t1 = time.time() - t0
        print(f"with narrowing took {t1} seconds")
    else:
        assert len(pcs_to_consider) > 0, "can't consider all points without narrowing"
        pcs_in_region = find.filter_point_codes_in_region_one_by_one(pcs_to_consider, center_pc, d_gc)
    return pcs_in_region


def get_point_codes_from_file(fp):
    with open(fp) as f:
        lines = f.readlines()
    pcs = [l.strip() for l in lines]
    icm.verify_valid_point_codes(pcs)
    return pcs


def make_point_code_file_for_random_region(db):
    # min_pc_iterations = random.randint(0, 8)
    # max_pc_iterations = min_pc_iterations + random.randint(0, 2)
    # expected_pc_iterations = (min_pc_iterations + max_pc_iterations) / 2
    # pc = icm.get_random_point_code(min_pc_iterations, expected_pc_iterations, max_pc_iterations)
    center_pc = random.choice(db.df.index)  # use a point we know is in the database
    pc_iterations = icm.get_iteration_number_from_point_code(center_pc)
    # resolution_iterations = pc_iterations + random.randint(2, 4)
    d_gc = np.random.uniform(0.02, 0.15)
    return make_point_code_file_for_region(db, center_pc, d_gc)


def make_point_code_file_for_region(db, center_pc, d_gc, parent_dir="PointFiles", pcs_to_consider=None, use_narrowing=True, overwrite=False):
    fname_prefix = "pcs_in_db"
    fp = get_point_code_filename(center_pc, d_gc, fname_prefix, parent_dir)
    if (not overwrite) and os.path.exists(fp):
        print(f"file {fp} exists, not going to calculate this region")
        return

    pcs_in_region = get_point_codes_in_database_in_region(db, center_pc, d_gc, use_narrowing=use_narrowing, pcs_to_consider=pcs_to_consider)
    write_point_codes_to_file(pcs_in_region, center_pc, d_gc, fname_prefix, parent_dir=parent_dir)


def write_point_codes_to_file(pcs_in_region, center_pc, d_gc, prefix_str, parent_dir="PointFiles"):
    pc_fp = get_point_code_filename(center_pc, d_gc, prefix_str, parent_dir)
    with open(pc_fp, "w") as f:
        for center_pc in pcs_in_region:
            f.write(center_pc + "\n")
    print(f"point codes written to {pc_fp}")


def point_code_file_exists(center_pc, d_gc, prefix_str, parent_dir=None):
    fp = get_point_code_filename(center_pc, d_gc, prefix_str, parent_dir)
    return os.path.exists(fp)


def get_point_code_filename(center_pc, d_gc, prefix_str, parent_dir=None):
    fname = f"{prefix_str}_{center_pc}_{d_gc}.txt"
    if parent_dir is None:
        fp = fname
    else:
        fp = os.path.join(parent_dir, fname)
    return fp


def get_point_code_file_regex():
    return "pcs_in_db_(?P<date>[\d\-]+)_(?P<pc>[A-L][0123]*)_(?P<d>[.\d]+).txt"


def get_random_point_code_file(pc_dir):
    pattern = get_point_code_file_regex()
    fnames = [x for x in os.listdir(pc_dir) if re.match(pattern, x)]
    fname = random.choice(fnames)
    fp = os.path.join(pc_dir, fname)
    return fname, fp


def get_point_code_and_distance_from_filename(fname):
    pattern = get_point_code_file_regex()
    match = re.match(pattern, fname)
    center_pc = match.group("pc")
    region_radius_gc = float(match.group("d"))
    return center_pc, region_radius_gc


def partition_into_max_sum_groups(ints, max_sum):
    # e.g. if you get a list like [4, 9, 5, 12, 1, 2, 3] with max_sum 15
    # then use greedy approach to put them into the first "box" they'll fit in
    # 4 and 9 go in first box, making its sum 13
    # 5 can't fit so it goes in a new box (2nd)
    # 12 can't fit in either box so it goes in a new box (3rd)
    # 1 can fit in the first box, 2 and 3 can fit in the 2nd box
    # result: [4, 9, 1], [5, 2, 3], [12]
    boxes = []
    for n in ints:
        placed = False
        for box in boxes:
            s = sum(box)
            fits = s + n <= max_sum
            if fits:
                box.append(n)
                placed = True
                break
        if not placed:
            # it didn't fit anywhere, make a new box
            new_box = [n]
            boxes.append(new_box)
    return boxes


def get_prefix_lookup_numbers_from_point_codes(pcs):
    num_to_pcs = {}  # for checking for duplicates
    res = []
    duplicates_found = False
    for pc in pcs:
        n = get_prefix_lookup_number_from_point_code(pc)
        if n in num_to_pcs:
            print(f"{pc=} created duplicate lookup number {n}, also found with pcs {num_to_pcs[n]}")
            num_to_pcs[n].append(pc)
            duplicates_found = True
        else:
            num_to_pcs[n] = [pc]
        res.append(n)
    if duplicates_found:
        assert all(len(set(icm.strip_trailing_zeros(pc) for pc in these_pcs)) == 1 for num, these_pcs in num_to_pcs.items())
        raise RuntimeError("duplicate lookup numbers created; see above")
    return res


def get_prefix_lookup_number_from_point_code(pc):
    # reverse the number, treat the point letter as the least significant digit (base 12)
    # all other digits are base 4
    # so we can tell if the string starts with a prefix
    # because it will be congruent to that prefix mod some base
    if pc[0] == "A":
        assert all(x == "0" for x in pc[1:])
        return -2
    elif pc[0] == "B":
        assert all(x == "0" for x in pc[1:])
        return -3
    d1 = {y:x for x,y in enumerate("CDEFGHIJKL")}
    d2 = {x: int(x) for x in "0123"}  # so it will KeyError for other values
    n = d1[pc[0]] + 10 * sum(4**(p-1) * d2[pc[p]] for p in range(1, len(pc)))
    if n > IcosahedronPointDatabase.MAX_INT_VALUE:
        raise ValueError(f"{pc} has lookup number {n}, which exceeds max int64 value of {IcosahedronPointDatabase.MAX_INT_VALUE}")
    return n


def get_prefix_lookup_modulus(prefix):
    return 1 if len(prefix) == 0 else 10 * 4**(len(prefix)-1)


def lookup_number_matches_prefix_number(lookup_number, modulus, prefix_number):
    if prefix_number == -2:  # A
        return lookup_number == -2
    elif prefix_number == -3:  # B
        return lookup_number == -3
    return lookup_number % modulus == prefix_number


def get_mask_point_codes_starting_with_prefix_using_lookup_number(lookup_numbers, prefix):
    prefix_num = get_prefix_lookup_number_from_point_code(prefix)
    modulus = get_prefix_lookup_modulus(prefix)
    matches_mask = lookup_number_matches_prefix_number(lookup_numbers, modulus, prefix_num)
    return matches_mask


def get_point_codes_starting_with_prefix_using_lookup_number(df, lookup_numbers, prefix):
    matches_mask = get_mask_point_codes_starting_with_prefix_using_lookup_number(lookup_numbers, prefix)
    pcs = df.index[matches_mask]
    return pcs


def check_no_trailing_zeros_in_point_codes(pcs):
    for pc in pcs:
        assert pc[-1] != "0", f"can't add point code with trailing zeros to database, {pc=}"


def initialize_default_value_dataframe_from_control_points(db_root_dir):
    # might need to use this later once more variables are added as control point images
    control_data_fp = os.path.join(db_root_dir, "control_data.h5")
    data_fp = os.path.join(db_root_dir, "data.h5")
    df = pd.read_hdf(control_data_fp).copy(deep=True)
    default_values = {
        "elevation": get_default_values_of_conditions("Cada II", "elevation"),
        "volcanism": get_default_values_of_conditions("Cada II", "volcanism")
    }
    df.insert(list(df.columns).index("elevation_condition")+1, "elevation", [default_values["elevation"][x] for x in df["elevation_condition"]])
    df.insert(list(df.columns).index("volcanism_condition")+1, "volcanism", [default_values["volcanism"][x] for x in df["volcanism_condition"]])
    print(df)
    df.to_hdf(data_fp, "data")
    print("wrote default values to data.h5")



if __name__ == "__main__":  
    db_root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"

    # to start the database over based on just the control point images
    initialize_default_value_dataframe_from_control_points(db_root_dir)
    sys.exit()

    db = IcosahedronPointDatabase.load(db_root_dir)
    df = db.df

    # db.write_as_images()  # experimental, not working yet

    pc_dir = "PointFiles"
    fname, fp = get_random_point_code_file(pc_dir)
    center_pc, region_radius_gc = get_point_code_and_distance_from_filename(fname)

    max_iterations = 10
    
    if point_code_file_exists(center_pc, region_radius_gc, "all_pcs"):
        print("file exists, not calculating this region")
    control_pcs = get_point_codes_from_file(fp)
    all_pcs = icm.get_region_around_point_code_by_spreading(center_pc, region_radius_gc, max_iterations)
    new_fname_prefix = f"pcs_iter{max_iterations}"
    pu.scatter_icosa_points_by_code(all_pcs, show=True)
    write_point_codes_to_file(all_pcs, center_pc, region_radius_gc, new_fname_prefix)


    # random.shuffle(fps)
    # variable_index = 0
    # for fp in fps:
    #     print(fp)
    #     pcs = get_point_codes_from_file(fp)
    #     pu.plot_variable_at_point_codes(pcs, db, variable_index)
    #     plt.show()

    # while True:
    #     make_point_code_file_for_random_region(df)

    # direct filtering one by one is still really slow (over an hour for one test of reasonable size)
    # whereas combination of narrowing and filtering the split region is a few minutes
    # t0 = time.time()
    # print("\n----\nnow just filtering directly")
    # pcs_in_region = find.filter_point_codes_in_region_one_by_one(df.index, pc, d_gc)
    # print(f"there are {in_region_mask.sum()} points in the region")
    # print(list(pcs_in_region))
    # t2 = time.time() - t0
    # print(f"with narrowing took {t1} seconds")
    # print(f"direct filtering took {t2} seconds")
