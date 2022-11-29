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
import random
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import IcosahedronMath as icm
from BiDict import BiDict
from PointCodeTrie import PointCodeTrie
import FindPointsInCircle as find
import PlottingUtil as pu


class IcosahedronPointDatabase:
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
        db.variables_file = os.path.join(root_dir, "variables.txt")
        db.metadata_file = os.path.join(root_dir, "metadata.txt")

        db.variables_dict = BiDict(int, str)
        db.metadata = {
            "n_point_code_chars_per_level": n_point_code_chars_per_level,
        }
        db.cache = {}
        touch(db.variables_file)
        db.write_metadata()
        return db

    @staticmethod
    def load(root_dir):
        print(f"loading database from {root_dir}")
        db = IcosahedronPointDatabase()
        db.root_dir = root_dir
        db.variables_file = os.path.join(root_dir, "variables.txt")
        db.metadata_file = os.path.join(root_dir, "metadata.txt")
        db.data_file = os.path.join(root_dir, "data.h5")
        db.variables_dict = IcosahedronPointDatabase.get_variables_dict_from_file(db.variables_file)
        db.metadata = IcosahedronPointDatabase.get_metadata_from_file(db.metadata_file)
        db.cache = {}
        db.read_hdf()
        print(f"done loading database from {root_dir}")
        return db

    @staticmethod
    def db_exists(root_dir):
        if not os.path.exists(root_dir):
            return False
        contents = os.listdir(root_dir)
        necessary_contents = ["IcosahedronPointDatabase.txt", "variables.txt", "metadata.txt", "blocks"]  # dirs don't end with slash in the listing
        found = [x in contents for x in necessary_contents]
        if all(found):
            return True
        elif not any(found):
            return False
        else:
            raise Exception(f"database partially exists: {root_dir}")

    def add_variable(self, variable_name):
        var_dict = self.get_variables_dict()
        if variable_name in var_dict.keys(str, int):
            print("cannot add existing variable", variable_name)
            return
        indexes = var_dict.keys(int, str)
        if len(indexes) == 0:
            new_index = 0
        else:
            new_index = max(indexes) + 1
        var_dict[new_index] = variable_name
        self.variables_dict = var_dict
        self.write_variables_dict()

    def write_variables_dict(self):
        lines = []
        for index, name in sorted(self.variables_dict.items(int, str)):
            l = f"{index}:{name}"
            lines.append(l)
        s = "\n".join(lines)
        with open(self.variables_file, "w") as f:
            f.write(s)

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

    def read_hdf(self):
        self.df = pd.read_hdf(self.data_file)

    def get_variables_dict(self):
        if self.variables_dict is not None:
            return self.variables_dict
        else:
            return IcosahedronPointDatabase.get_variables_dict_from_file(self.variables_file)

    @staticmethod
    def get_variables_dict_from_file(fp):
        with open(fp) as f:
            lines = f.readlines()
        d = {}
        for l in lines:
            index, name = l.strip().split(":")
            index = int(index)
            d[index] = name
        return BiDict.from_dict(d)

    @staticmethod
    def get_metadata_from_file(fp):
        with open(fp) as f:
            lines = f.readlines()
        d = {}
        for l in lines:
            var, val = l.strip().split(":")
            d[var] = int(val)
        return d

    def get_single_point(self, pn, variable_name):
        if pn in self.cache:
            return self.cache[pn].get(variable_name)
        else:
            return self.get_single_point_from_file(pn, variable_name)

    def get_single_point_multiple_variables(self, pn, variable_names):
        if pn in self.cache:
            d = self.cache[pn]
        else:
            d = self.get_single_point_all_variables_from_file(pn)
        return {vn: d.get(vn) for vn in variable_names}

    def get_multiple_points(self, pns, variable_name):
        pn_set = set(pns)
        cached_pns = set(self.cache.keys()) & pn_set
        cached_pns = set(pn for pn in cached_pns if variable_name in self.cache[pn])  # if this var not found, read from file (because sometimes the point is only cached with some variables but not others)
        non_cached_pns = pn_set - cached_pns
        d = {}
        for pn in cached_pns:
            d[pn] = self.cache[pn].get(variable_name)
        d_from_file = self.get_multiple_points_from_file(non_cached_pns, variable_name)
        d.update(d_from_file)
        return d

    def get_multiple_points_multiple_variables(self, pns, variable_names):
        raise NotImplementedError

    def get_single_point_from_file(self, pn, variable_name):
        d = self.get_single_point_all_variables_from_file(pn)
        variable_number = self.get_variable_number_from_name(variable_name)
        return d.get(variable_name)

    def get_single_point_all_variables(self, pn):
        if pn in self.cache:
            return self.cache[pn]
        else:
            return self.get_single_point_all_variables_from_file(pn)
        
    def get_single_point_all_variables_from_file(self, pn):
        block_fp = self.get_block_fp_for_point_number(pn)
        if not os.path.exists(block_fp):
            # raise KeyError(f"no data for point {pn}")
            return None
        with open(block_fp) as f:
            lines = f.readlines()

        # find the line starting with this point, if any
        block_size = self.metadata["block_size"]
        block_number = pn // block_size
        block_start = block_number * block_size
        adjusted_pn = pn - block_start
        assert 0 <= adjusted_pn < block_size, (pn, block_size, block_number, block_start, adjusted_pn)
        lines = [l.strip().split(",") for l in lines]
        lines_this_point = [l for l in lines if int(l[0]) == adjusted_pn]
        if len(lines_this_point) == 0:
            # raise KeyError(f"no data for point {pn}")
            return None
        elif len(lines_this_point) > 1:
            raise RuntimeError(f"point {pn} found more than once!")
        else:
            l, = lines_this_point
            d = IcosahedronPointDatabase.get_all_variables_from_line(l)
            self.add_to_cache(pn, d)
            return d

    def get_multiple_points_from_file(self, pns, variable_name):
        variable_number = self.get_variable_number_from_name(variable_name)
        block_size = self.metadata["block_size"]
        block_number_to_pns = {}
        for pn in pns:
            block_number = pn // block_size
            if block_number not in block_number_to_pns:
                block_number_to_pns[block_number] = []
            block_number_to_pns[block_number].append(pn)
        block_number_to_fp = {bn: self.get_block_fp_for_block_number(bn) for bn in block_number_to_pns.keys()}
        res_by_pn = {}
        for block_number, pns_this_block in block_number_to_pns.items():
            block_fp = block_number_to_fp[block_number]
            block_start = block_number * block_size
            adjusted_pns_this_block = [pn - block_start for pn in pns_this_block]
            assert all(0 <= apn < block_size for apn in adjusted_pns_this_block), "bad adjusted point number"
            with open(block_fp) as f:
                lines = f.readlines()
            lines = [l.strip().split(",") for l in lines]
            line_dict = {int(l[0]): l for l in lines}
            for apn in adjusted_pns_this_block:
                pn = apn + block_start
                l = line_dict.get(apn)
                if l is None:
                    res_by_pn[pn] = None
                else:
                    val = IcosahedronPointDatabase.get_variable_from_line(l, variable_number)
                    self.add_to_cache(pn, variable_name, val)
                    res_by_pn[pn] = val
        return res_by_pn

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
    def get_variable_from_line(l, variable_number):
        d = IcosahedronPointDatabase.get_all_variables_from_line(l)
        val = d.get(variable_number)
        # typ = self.get_variable_type_from_name(variable_name)
        # for now, make everything in the db an int, can capture enums, bools, and floats to some precision, and that way I don't have to parse a file to figure out what the types are supposed to be; put units in the varname if you care about that, e.g. elevation_meters
        return val

    def add_to_cache(self, pn, variable_dict):
        self.cache[pn] = variable_dict

    def set_single_point(self, pn, variable_name, value, write=False):
        check_int(value)
        variable_number = self.get_variable_number_from_name(variable_name)
        self.add_to_cache(pn, variable_name, value)
        if write:
            self.write()  # don't do this too often or it will be slow

    def set_multiple_points(self, pns, variable_name, values, write=False):
        variable_number = self.get_variable_number_from_name(variable_name)
        for pn, val in zip(pns, values):
            check_int(val)
            self.add_to_cache(pn, variable_name, val)
        if write:
            self.write()

    # def get_all_point_numbers_with_data(self):
    #     print("getting points with data in database")
    #     block_numbers = self.get_all_block_numbers_with_data()
    #     res = set()
    #     for block_number in block_numbers:
    #         # print(f"reading block {block_number}")
    #         pns = self.get_point_numbers_with_data_in_block(block_number)
    #         assert res & pns == set(), "overlap"
    #         res |= pns
    #     print("done getting points with data")
    #     return res

    # def get_point_numbers_with_data_in_block(self, block_number):
    #     block_size = self.metadata["block_size"]
    #     block_start = block_number * block_size
    #     block_fp = self.get_block_fp_for_block_number(block_number)
    #     with open(block_fp) as f:
    #         lines = f.readlines()
    #     res = set()
    #     for l in lines:
    #         l = l.strip().split(",")
    #         adjusted_pn = int(l[0])
    #         pn = block_start + adjusted_pn
    #         assert pn not in res
    #         # print(f"got point number {pn}")
    #         res.add(pn)
    #     return res

    # def get_all_block_numbers_with_data(self):
    #     block_files = os.listdir(self.blocks_dir)
    #     res = set()
    #     for fname in block_files:
    #         block_number = int(fname.split("_")[0].replace("Block", ""))
    #         assert block_number not in res
    #         res.add(block_number)
    #     return res

    # def get_int_from_line_label(self, line_label):
    #     N = self.metadata["n_point_code_chars_per_level"]
    #     assert 1 <= len(line_label) <= N, repr(line_label)
    #     # base-4 number
    #     # note that if it is less than N chars long, it has trailing zeros, not leading
    #     # so 1 acts like 100 and 11 acts like 110 (for N=3)
    #     to_n = lambda c: "0123".index(c)
    #     x = 0
    #     for i, c in enumerate(line_label):
    #         n = to_n(c)
    #         x += n * (4 ** ((N-1)-i))
    #     # print(f"{line_label=} gave {x=}")
    #     return x

    # def get_filepath_and_line_label_for_point_code(self, pc):
    #     root_dir = self.root_dir
    #     N = self.metadata["n_point_code_chars_per_level"]
    #     parent_dir = os.path.join(root_dir, "data_by_point_code/")
    #     head = pc[0]
    #     tail = pc[1:]
    #     assert head in "ABCDEFGHIJKL"
    #     n_subdirs = (len(tail) - 1) // N
    #     chars_remaining = len(tail)
    #     subdirs = []
    #     for i in range(n_subdirs):
    #         digits = tail[N*i : N*i+N]
    #         subdirs.append(digits)
    #         chars_remaining -= N
    #     filename = "data.h5"
    #     line_label = tail[-chars_remaining:]
    #     fp = os.path.join(parent_dir, head, *subdirs, filename)
    #     return fp, line_label

    def get_variable_number_from_name(self, name):
        return self.variables_dict[name]

    def get_variable_name_from_number(self, number):
        return self.variables_dict[number]

    def __getitem__(self, tup):
        pns, variable_name = tup
        if type(pns) is int:
            pn = pns
            return self.get_single_point(pn, variable_name)
        else:
            return self.get_multiple_points(pns, variable_name)

    def __setitem__(self, tup, val):
        pns, variable_name = tup
        if type(pns) is int:
            pn = pns
            self.set_single_point(pn, variable_name, val)
        else:
            self.set_multiple_points(pns, variable_name, val)

    def add_values(self, pns, varname, vals):
        # instead of db[pns, varname] += vals, since that won't work with the dictionary return value and I can't do __iadd__ on the database object itself because I'm not saying db += vals
        current_vals = self[pns, varname]
        if type(pns) is int:
            pns = [pns]
        if type(vals) is int:
            vals = [vals]
        assert len(pns) == len(vals)
        new_vals = []
        for pn, val in zip(pns, vals):
            new_val = current_vals[pn] + val
            new_vals.append(new_val)
        self[pns, varname] = new_vals

    def get_mask_point_codes_with_prefix(self, prefix, pandas_method=True):
        # want to return true for things like "D" starting with "D00"

        if pandas_method:
            return self.df.index.str.ljust(len(prefix), "0").str.startswith(prefix)
        else:
            return self.get_mask_point_codes_with_prefixes([prefix], pandas_method=False)

    def get_all_point_codes_with_prefix(self, prefix):
        mask = self.get_mask_point_codes_with_prefix(prefix)
        return self.df.index[mask]

    def get_mask_point_codes_with_prefixes(self, prefixes, pandas_method=True):
        print(f"creating point code prefix mask")
        mask = pd.Series([False] * len(self.df.index), index=self.df.index)
        # without setting the mask's index to be the same as the df's it will error

        # times for using pandas method vs manually are very similar (e.g. 86 s and 87 s)
        if pandas_method:
            for prefix in prefixes:
                mask |= self.get_mask_point_codes_with_prefix(prefix, pandas_method=True)
        else:
            max_prefix_len = max(len(x) for x in prefixes)
            for pc in self.df.index:
                pc = pc.ljust(max_prefix_len, "0")
                for prefix in prefixes:
                    if pc.startswith(prefix):
                        mask[pc] = True
                        break

        print(f"done creating point code prefix mask, has {mask.sum()} items")
        return mask

    def clear_cache(self):
        self.cache = {}
        print("db cache cleared")

    def write(self, clear_cache=True):
        # update the block files on disk
        block_size = self.metadata["block_size"]
        blocks_in_cache = set(pn // block_size for pn in self.cache.keys())
        for block_number in blocks_in_cache:
            print("writing block_number", block_number)
            block_start = block_number * block_size
            pns = set([p for p in self.cache.keys() if p // block_size == block_number])
            fps = set(self.get_block_fp_for_point_number(p) for p in pns)
            assert len(fps) == 1, "problem getting block number from point numbers"
            fp, = list(fps)
            # for each point, read what's in the file into a dict, update the dict with what's in the cache (but don't delete stuff that's there but isn't in the cache), save line as string, then write them all to the file
            data_on_disk = {}
            if os.path.exists(fp):
                with open(fp) as f:
                    lines = f.readlines()
                for l in lines:
                    l_split = l.strip().split(",")
                    adjusted_p_i = int(l_split[0])
                    p_i = adjusted_p_i + block_start
                    d = {}
                    for item in l_split[1:]:
                        k,v = item.split("=")
                        k = int(k)
                        v = int(v)
                        d[k] = v
                    data_on_disk[p_i] = d
            # if fp doesn't exist then just leave data_on_disk empty

            lines_to_write = []
            all_points_to_write = sorted(set(data_on_disk.keys()) | set(pns))
            for p in all_points_to_write:
                if p in data_on_disk:
                    d = data_on_disk[p]
                else:
                    d = {}
                if p in self.cache:
                    for var, val in self.cache[p].items():
                        var_num = self.get_variable_number_from_name(var)
                        d[var_num] = val
                assert len(d) > 0
                adjusted_pn = p - block_start
                assert 0 <= adjusted_pn < block_size, (p, block_size, block_number, block_start, adjusted_pn)
                new_l = IcosahedronPointDatabase.get_line_from_dict(adjusted_pn, d)
                lines_to_write.append(new_l)
            with open(fp, "w") as f:
                for l in lines_to_write:
                    f.write(l + "\n")
        print("db written")
        if clear_cache:
            self.clear_cache()

    @staticmethod
    def get_line_from_dict(line_label, d):
        items = [line_label]
        for k,v in sorted(d.items()):
            items.append(f"{k}={v}")
        return ",".join(items)
    
    def write_old_block_format_to_hdf5(self):
        pns = list(db.get_all_point_numbers_with_data())
        var_dict = db.get_variables_dict()
        variable_indices = sorted(var_dict.keys(int, str))

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
    if type(value) is not int:
        raise TypeError(f"Database only accepts int values (you gave {value} of type {type(value)}).\nIf you want enum, make them int shorthands in the condition_array_dir.\nIf you want floats, choose the precision you want and make ints of that, e.g. elevation in millimeters.")


def get_point_codes_from_file(fp):
    with open(fp) as f:
        lines = f.readlines()
    pcs = [l.strip() for l in lines]
    icm.verify_valid_point_codes(pcs)
    return pcs


def make_point_code_file_for_random_region(df):
    # min_pc_iterations = random.randint(0, 8)
    # max_pc_iterations = min_pc_iterations + random.randint(0, 2)
    # expected_pc_iterations = (min_pc_iterations + max_pc_iterations) / 2
    # pc = icm.get_random_point_code(min_pc_iterations, expected_pc_iterations, max_pc_iterations)
    center_pc = random.choice(df.index)  # use a point we know is in the database
    pc_iterations = icm.get_iteration_number_from_point_code(center_pc)
    resolution_iterations = pc_iterations + random.randint(2, 4)
    narrowing_iterations = 5 #min(resolution_iterations - 1, random.randint(0, 3))
    d_gc = np.random.uniform(0.02, 0.15)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    fname_prefix = f"pcs_in_db_{today}"
    if point_code_file_exists(center_pc, d_gc, fname_prefix):
        print("file exists, not going to calculate this region")
        return

    # use narrowing to get which watersheds are all inside, all outside, and split
    t0 = time.time()
    inside, outside, split = icm.narrow_watersheds_by_distance(center_pc, d_gc, narrowing_iterations)
    print("inside", inside)
    print("outside", outside)
    print("split", split)

    # now get point codes that are in each of these
    # label the inside ones as True, the outside ones as False
    # then filter rows by prefix to get the split ones and check them one by one
    # (see how many there are to check as well, how long this will take)
    df["in_region"] = np.nan
    df["in_region"] = df["in_region"].astype("boolean")
    inside_mask = db.get_mask_point_codes_with_prefixes(inside)
    outside_mask = db.get_mask_point_codes_with_prefixes(outside)
    split_mask = (~inside_mask) & (~outside_mask)
    df.loc[inside_mask, "in_region"] = True
    df.loc[outside_mask, "in_region"] = False
    split_pcs = df.index[split_mask]
    split_pcs_in_region = find.filter_point_codes_in_region_one_by_one(split_pcs, center_pc, d_gc)
    for pc1 in split_pcs:
        if pc1 in split_pcs_in_region:
            df.loc[pc1, "in_region"] = True
        else:
            df.loc[pc1, "in_region"] = False
    print(df)
    in_region_mask = df["in_region"]
    print(f"there are {in_region_mask.sum()} points in the region")
    pcs_in_region = df.loc[in_region_mask].index
    t1 = time.time() - t0
    print(f"with narrowing took {t1} seconds")

    write_point_codes_to_file(pcs_in_region, center_pc, d_gc, fname_prefix, parent_dir="PointFiles")
    df.drop(["in_region"], axis=1)


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



if __name__ == "__main__":
    root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"
    db = IcosahedronPointDatabase.load(root_dir)
    df = db.df

    pc_dir = "PointFiles"
    pattern = "pcs_in_db_(?P<date>[\d\-]+)_(?P<pc>[A-L][0123]*)_(?P<d>[.\d]+).txt"
    fnames = [x for x in os.listdir(pc_dir) if re.match(pattern, x)]
    fname = random.choice(fnames)
    print(fname)
    fp = os.path.join(pc_dir, fname)
    match = re.match(pattern, fname)
    center_pc = match.group("pc")
    region_radius_gc = float(match.group("d"))
    if point_code_file_exists(center_pc, region_radius_gc, "all_pcs"):
        print("file exists, not calculating this region")
    control_pcs = get_point_codes_from_file(fp)
    max_iterations = max(len(pc) for pc in control_pcs) - 1
    all_pcs = icm.get_region_around_point_code_by_spreading(center_pc, region_radius_gc, max_iterations)
    write_point_codes_to_file(all_pcs, center_pc, region_radius_gc, "all_pcs")



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
