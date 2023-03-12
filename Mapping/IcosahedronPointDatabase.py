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
import json

import IcosahedronMath as icm
from BiDict import BiDict
import FindPointsInCircle as find
import PlottingUtil as pu
from LoadMapData import get_default_values_of_conditions, translate_array_by_dict, create_control_point_dataframe_from_images
from XyzLookupAncestryGraph import XyzLookupAncestryGraph


class IcosahedronPointDatabase:
    VALUE_DTYPE = np.dtype("int64")
    CONDITION_DTYPE = np.dtype("int8")
    COLUMN_VALUE_LIMITS = {
        np.dtype("int8"): [-2**7, 2**7-1],
        np.dtype("int64"): [-2**63, 2**63-1],
    }
    MAX_INT64_VALUE = 2**63-1
    SPECIAL_COLUMN_NAMES = ["index", "prefix_lookup_number"]

    def __init__(self):
        pass

    @staticmethod
    def new(root_dir, world_name):
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
        db.metadata_fp = get_metadata_fp(root_dir)

        db.variables_dict = BiDict(int, str)
        db.metadata = {
            "world_name": world_name,
        }
        db.write_metadata()
        return db

    @staticmethod
    def load(root_dir):
        print(f"loading database from {root_dir}")
        db = IcosahedronPointDatabase()
        db.root_dir = root_dir
        db.metadata_fp = get_metadata_fp(root_dir)
        db.data_file = get_data_fp(root_dir)
        db.metadata = IcosahedronPointDatabase.get_metadata_from_file(db.metadata_fp)
        db.read_hdf()
        print(f"-- done loading database from {root_dir}")
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
        json.dump(d, self.metadata_fp)

    def write_hdf(self):
        self.coerce_dtypes()
        self.validate()
        self.df.to_hdf(self.data_file, key="data")
        print("IcosahedronPointDatabase written to file.\n")

    def read_hdf(self):
        self.df = pd.read_hdf(self.data_file)
        self.validate()

    def validate(self):
        self.verify_df_dtypes()
        self.check_no_duplicates_in_index()
        self.check_no_na()
    
    def check_no_duplicates_in_index(self):
        assert self.df.index.has_duplicates is False
        # don't do `not object.attr` because that will be falsy if the attribute is absent
    
    def check_no_na(self):
        assert not pd.isna(self.df.index).any()
        for col in self.df.columns:
            assert not pd.isna(self.df[col]).any(), f"NA value in column {col}"

    def get_variables(self):
        return sorted(self.df.columns)

    def get_n_variables(self):
        return len(self.get_variables())
    
    @staticmethod
    def is_condition_variable(variable_name):
        # things like the enumerated type of elevation_condition (ocean, coast, land, etc.)
        res = variable_name.endswith("_condition")
        if res:
            assert not IcosahedronPointDatabase.is_special_column(variable_name), "shouldn't have special variables ending in '_condition'"
        return res
    
    @staticmethod
    def is_value_variable(variable_name):
        # things like the amount of actual elevation at a point
        is_condition = IcosahedronPointDatabase.is_condition_variable(variable_name)
        is_special = IcosahedronPointDatabase.is_special_column(variable_name)
        return (not is_condition) and (not is_special)
    
    @staticmethod
    def is_special_column(variable_name):
        # things like the row indices or info about the points themselves,
        # rather than map variables
        return variable_name in IcosahedronPointDatabase.SPECIAL_COLUMN_NAMES
    
    def get_condition_variables(self):
        return sorted(x for x in self.df.columns if IcosahedronPointDatabase.is_condition_variable(x))
    
    def get_value_variables(self):
        return sorted(x for x in self.df.columns if IcosahedronPointDatabase.is_value_variable(x))

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
            d = json.load(f)
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
        is_condition_column = IcosahedronPointDatabase.is_condition_variable(variable_name)
        # check_int(value, is_condition_column)
        self.df.loc[pc, variable_name] = value
        # self.validate()

    def set_multiple_points(self, pcs, variable_name, values):
        is_condition_column = IcosahedronPointDatabase.is_condition_variable(variable_name)
        # for val in values:
        #     check_int(val, is_condition_column)
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
        lns = icm.get_prefix_lookup_numbers_from_point_codes(pcs)
        for ln in lns:
            if ln not in self.df.index:
                missing.append(ln)
        if len(missing) > 0:
            print("these points are missing from the database:", missing)
            raise KeyError("missing points; see list above")

    def coerce_dtypes(self):
        lst = []
        for colname in self.get_condition_variables():
            dtype = IcosahedronPointDatabase.CONDITION_DTYPE
            lst.append([colname, dtype])
        for colname in self.get_value_variables():
            dtype = IcosahedronPointDatabase.VALUE_DTYPE
            lst.append([colname, dtype])

        for colname, dtype in lst:
            min_val, max_val = IcosahedronPointDatabase.COLUMN_VALUE_LIMITS[dtype]
            vals = self.df[colname]
            assert (min_val <= vals).all(), f"{colname} outside minimum value"
            assert (vals <= max_val).all(), f"{colname} outside maximum value"
            self.df[colname] = vals.astype(dtype)

    def verify_df_dtypes(self):
        any_error = False

        for colname in self.get_condition_variables():
            dtype = self.df[colname].dtype
            okay = dtype is IcosahedronPointDatabase.CONDITION_DTYPE
            if not okay:
                print(f"dtype of condition column {colname} is {dtype}, should be {IcosahedronPointDatabase.CONDITION_DTYPE}")
                any_error = True

        for colname in self.get_value_variables():
            dtype = self.df[colname].dtype
            okay = dtype is IcosahedronPointDatabase.VALUE_DTYPE
            if not okay:
                print(f"dtype of value column {colname} is {dtype}, should be {IcosahedronPointDatabase.VALUE_DTYPE}")
                any_error = True

        if any_error:
            raise TypeError("df dtypes wrong; see above")

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
            raise NotImplementedError
        # im = Image.fromarray(arr)
        raise NotImplementedError

    def get_all_lookup_numbers(self):
        return self.df.index

    def get_mask_point_codes_with_prefix(self, prefix):
        # want to return true for things like "D" starting with "D00"
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

        lookup_numbers = np.array(self.df.index)  # pd.Series is WAY slower somehow
        for i, prefix in enumerate(prefixes):
            this_mask = get_mask_point_codes_starting_with_prefix_using_lookup_number(lookup_numbers, prefix)
            mask = mask | this_mask  # this line takes forever for pd.Series but fast for np.array
        print("-- done creating point code prefix mask")
        return mask

    @staticmethod
    def get_line_from_dict(line_label, d):
        items = [line_label]
        for k,v in sorted(d.items()):
            items.append(f"{k}={v}")
        return ",".join(items)
    
    def write_old_block_format_to_hdf5(self):
        raise Exception("should not have to use again")
        # pns = list(db.get_all_point_numbers_with_data())

        # if os.path.exists("data.h5"):
        #     df = pd.read_hdf("data.h5")
        #     n_points_in_file = (~df.index.duplicated()).sum()
        # else:
        #     n_points_in_file = 0
        # 
        # n_pns = len(pns)
        # print(f"{n_points_in_file=} / {n_pns=}")

        # df = pd.DataFrame(columns=variable_indices, dtype=int)  # blank df for adding points quicker
        # for i, pn in enumerate(pns):
        #     # if df in file has 1001 rows, last i written was 1000, 
        #     # so next i needs to be 1001, so skip i < n
        #     if i < n_points_in_file:
        #         continue
        #     if i % 100 == 0:
        #         print(f"{i}/{n_pns} points done")
        #         print(df)
        #     pc = icm.get_point_code_from_point_number(pn)
        #     variables = db.get_single_point_all_variables(pn)
        #     row_index = pc
        #     df.loc[row_index, variables.keys()] = variables.values()

        #     if i % 10000 == 0 or i >= n_pns - 1:
        #         df_in_file = pd.read_hdf("data.h5")
        #         df = pd.concat([df_in_file, df])
        #         df = df[~df.index.duplicated()]
        #         n_points_in_file = len(df.index)
        #         df.to_hdf("data.h5", key="df")
        #         print("wrote h5")
        #         print("total rows so far:", n_points_in_file)
        #         df = pd.DataFrame(columns=variable_indices, dtype=int)  # start clean one for next round
        # 
        # df_in_file = pd.read_hdf("data.h5")
        # assert len(df_in_file.index) == n_pns
        # print("process complete; all data is stored in data.h5")

    def update_control_conditions_from_images(self):
        control_df = create_control_point_dataframe_from_images(self.metadata["world_name"])
        lns_with_control = set(control_df.index)
        lns_with_data = set(self.df.index)
        lns_with_control_no_data = lns_with_control - lns_with_data
        lns_with_data_no_control = lns_with_data - lns_with_control
        print(f"{len(lns_with_control)} points have control conditions")
        print(f"{len(lns_with_data)} points have data")
        print(f"{len(lns_with_control_no_data)} points have control conditions but no data")
        print(f"{len(lns_with_data_no_control)} points have data but no control conditions")

        # any point that now has a control condition (from the images, not from what was saved in the df before)
        # - will have its condition set to that new one
        # - but its value will not be touched (let the elevation generation take care of that)
        # any other point, even if the df had a condition for it before, is set to having no condition (the interpolation will run again, or if there is no interpolation being done for conditions, then those points will just be conditionless and we'll rely on the nearby control points to keep them in line)
        for control_variable in control_df.columns:
            assert control_variable.endswith("_condition"), control_variable
            self.df[control_variable] = -1  # reset all of them to unknown condition
            self.df.loc[control_df.index, control_variable] = control_df[control_variable]  # set the ones with new conditions

        control_df.to_hdf(get_control_data_fp(self.root_dir), "control_data")
        self.write_hdf()

        print("- done updating control conditions in the database from images")



def touch(fp):
    assert not os.path.exists(fp), "cannot touch existing file"
    open(fp, "w").close()


def check_int(value, is_condition_column):
    dtype = IcosahedronPointDatabase.CONDITION_DTYPE if is_condition_column else IcosahedronPointDatabase.VALUE_DTYPE
    if type(value) is not dtype:
        raise TypeError(f"value should be of type {dtype} (you gave {value} of type {type(value)}).\nIf you want enum, make them int shorthands in the condition_array_dir.\nIf you want floats, choose the precision you want and make ints of that, e.g. elevation in millimeters.")


def get_lookup_numbers_in_database_in_region(db, center_pc, d_gc, xyzg, use_narrowing=True, lns_to_consider=None):
    # print(f"{center_pc=}")
    center_ln = icm.get_prefix_lookup_number_from_point_code(center_pc)
    df = db.df
    if use_narrowing:
        assert lns_to_consider is None, "not implemented"
        # use narrowing to get which watersheds are all inside, all outside, and split
        t0 = time.time()
        narrowing_iterations = 5 #min(resolution_iterations - 1, random.randint(0, 3))
        inside, outside, split = icm.narrow_watersheds_by_distance(center_pc, d_gc, narrowing_iterations, xyzg)
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
        split_lns = df.index[split_mask]
        split_pcs_indices = np.where(split_mask)
        # split_pcs = icm.get_point_codes_from_prefix_lookup_numbers(split_lns)
        split_pc_in_region_mask_from_split_pcs = find.get_mask_points_in_region(split_lns, center_ln, d_gc, xyzg)
        split_pc_in_region_mask_df_index_aligned = np.full((len(df.index),), False)
        split_pc_in_region_mask_df_index_aligned[split_pcs_indices] = split_pc_in_region_mask_from_split_pcs
        point_in_region_mask = inside_mask | split_pc_in_region_mask_df_index_aligned
        # for pc in split_pcs:
        #     if pc in split_pcs_in_region:
        #         point_in_region_mask.loc[pc] = True
            # else:
            #     # yes it CAN happen and it's okay; raise Exception("shouldn't happen if it was already filtered?")
        print(f"there are {point_in_region_mask.sum()} points in the region")
        lns_in_region = df.loc[point_in_region_mask].index
        t1 = time.time() - t0
        print(f"with narrowing took {t1} seconds")
    else:
        assert len(lns_to_consider) > 0, "can't consider all points without narrowing"
        pcs_in_region = find.filter_point_codes_in_region_one_by_one(lns_to_consider, center_pc, d_gc)
        lns_in_region = icm.get_prefix_lookup_numbers_from_point_codes(pcs_in_region)
    return lns_in_region


def get_point_codes_from_file(fp):
    with open(fp) as f:
        lines = f.readlines()
    pcs = [l.strip() for l in lines]
    icm.verify_valid_point_codes(pcs)
    return pcs


def make_point_code_file_for_random_region(db):
    raise Exception("deprecated")
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
    raise Exception("deprecated")
    fname_prefix = "pcs_in_db"
    fp = get_point_code_filename(center_pc, d_gc, fname_prefix, parent_dir)
    if (not overwrite) and os.path.exists(fp):
        print(f"file {fp} exists, not going to calculate this region")
        return

    pcs_in_region = get_lookup_numbers_in_database_in_region(db, center_pc, d_gc, use_narrowing=use_narrowing, lns_to_consider=pcs_to_consider)
    write_point_codes_to_file(pcs_in_region, center_pc, d_gc, fname_prefix, parent_dir=parent_dir)


def write_point_codes_to_file(pcs_in_region, center_pc, d_gc, prefix_str, parent_dir="PointFiles"):
    raise Exception("deprecated")
    pc_fp = get_point_code_filename(center_pc, d_gc, prefix_str, parent_dir)
    with open(pc_fp, "w") as f:
        for center_pc in pcs_in_region:
            f.write(center_pc + "\n")
    print(f"point codes written to {pc_fp}")


def point_code_file_exists(center_pc, d_gc, prefix_str, parent_dir=None):
    raise Exception("deprecated")
    fp = get_point_code_filename(center_pc, d_gc, prefix_str, parent_dir)
    return os.path.exists(fp)


def get_point_code_filename(center_pc, d_gc, prefix_str, parent_dir=None):
    raise Exception("deprecated")
    fname = f"{prefix_str}_{center_pc}_{d_gc}.txt"
    if parent_dir is None:
        fp = fname
    else:
        fp = os.path.join(parent_dir, fname)
    return fp


def get_point_code_file_regex():
    raise Exception("deprecated")
    return "pcs_in_db_(?P<date>[\d\-]+)_(?P<pc>[A-L][0123]*)_(?P<d>[.\d]+).txt"


def get_random_point_code_file(pc_dir):
    raise Exception("deprecated")
    pattern = get_point_code_file_regex()
    fnames = [x for x in os.listdir(pc_dir) if re.match(pattern, x)]
    fname = random.choice(fnames)
    fp = os.path.join(pc_dir, fname)
    return fname, fp


def get_point_code_and_distance_from_filename(fname):
    raise Exception("deprecated")
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


def check_lookup_number_wont_overflow(pc):
    n = icm.get_prefix_lookup_number_from_point_code(pc)
    if n > IcosahedronPointDatabase.MAX_INT64_VALUE:
        raise ValueError(f"{pc} has lookup number {n}, which exceeds max int64 value of {IcosahedronPointDatabase.MAX_INT64_VALUE}")


def get_mask_point_codes_starting_with_prefix_using_lookup_number(lookup_numbers, prefix):
    check_lookup_number_wont_overflow(prefix)
    prefix_num = icm.get_prefix_lookup_number_from_point_code(prefix)
    modulus = icm.get_prefix_lookup_modulus(prefix)
    matches_mask = icm.lookup_number_matches_prefix_number(lookup_numbers, modulus, prefix_num)
    return matches_mask


def get_point_codes_starting_with_prefix_using_lookup_number(df, lookup_numbers, prefix):
    matches_mask = get_mask_point_codes_starting_with_prefix_using_lookup_number(lookup_numbers, prefix)
    lns = df.index[matches_mask]
    pcs = icm.get_point_codes_from_prefix_lookup_numbers(lns)
    return pcs


def check_no_trailing_zeros_in_point_codes(pcs):
    for pc in pcs:
        assert pc[-1] != "0", f"can't add point code with trailing zeros to database, {pc=}"


def get_control_data_fp(db_root_dir):
    return os.path.join(db_root_dir, "control_data.h5")


def get_data_fp(db_root_dir):
    return os.path.join(db_root_dir, "data.h5")


def get_metadata_fp(db_root_dir):
    return os.path.join(db_root_dir, "metadata.txt")


def initialize_default_value_dataframe_from_control_points(db_root_dir):
    # might need to use this later once more variables are added as control point images
    control_data_fp = get_control_data_fp(db_root_dir)
    data_fp = get_data_fp(db_root_dir)
    if os.path.exists(data_fp):
        raise Exception(f"database would be overwritten: {data_fp}")
    df = pd.read_hdf(control_data_fp).copy(deep=True)
    default_values = {
        "elevation": get_default_values_of_conditions("Cada II", "elevation"),
        "volcanism": get_default_values_of_conditions("Cada II", "volcanism")
    }
    df.insert(list(df.columns).index("elevation_condition")+1, "elevation", [default_values["elevation"][x] for x in df["elevation_condition"]])
    df.insert(list(df.columns).index("volcanism_condition")+1, "volcanism", [default_values["volcanism"][x] for x in df["volcanism_condition"]])
    print(df)
    df.to_hdf(data_fp, "data")
    print(f"wrote default values to {data_fp}")



if __name__ == "__main__":  
    db_root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"

    # to start the database over based on just the control point images
    # initialize_default_value_dataframe_from_control_points(db_root_dir)
    # sys.exit()

    db = IcosahedronPointDatabase.load(db_root_dir)
    df = db.df

    xyzg = XyzLookupAncestryGraph()
    pu.plot_variable_world_map_from_db(db, "elevation", xyzg, pixels_per_degree=10, show=True)

    # db.write_as_images()  # experimental, not working yet

    # pc_dir = "PointFiles"
    # fname, fp = get_random_point_code_file(pc_dir)
    # center_pc, region_radius_gc = get_point_code_and_distance_from_filename(fname)

    # max_iterations = 10
    
    # if point_code_file_exists(center_pc, region_radius_gc, "all_pcs"):
    #     print("file exists, not calculating this region")
    # control_pcs = get_point_codes_from_file(fp)
    # all_pcs = icm.get_region_around_point_code_by_spreading(center_pc, region_radius_gc, max_iterations)
    # new_fname_prefix = f"pcs_iter{max_iterations}"
    # pu.scatter_icosa_points_by_code(all_pcs, show=True)
    # write_point_codes_to_file(all_pcs, center_pc, region_radius_gc, new_fname_prefix)


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
