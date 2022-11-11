# store data indexed by icosahedral lattice point number
# multiple csvs by some block size e.g. 2**14
# all of them need the same header
# but need to allow sparseness, e.g. many points won't have values for certain variables, e.g. suppose "salt flat" is undefined most places but has values in small regions of the world, don't want to waste space having that defined as zero for almost every point
# maybe have a format that indexes the variables, every time a new variable is added it gets the next index number
# there is a global file telling which index means which variable, e.g. {0:is_land, 1:elevation, 2:volcanism, etc.}
# then the rows in the files look something like: 40128,0=0,1=-65.3,5=17.1,24=0
# so they're only showing the variables they're specified for


import os
import pathlib
import random
import pandas as pd

import IcosahedronMath as icm
from BiDict import BiDict


class IcosahedronPointDatabase:
    def __init__(self):
        pass

    @staticmethod
    def new(root_dir, block_size, n_point_code_chars_per_level):
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
        db.blocks_dir = os.path.join(root_dir, "blocks/")

        db.variables_dict = BiDict(int, str)
        db.metadata = {
            "block_size": block_size,
            "n_point_code_chars_per_level": n_point_code_chars_per_level,
        }
        db.cache = {}
        touch(db.variables_file)
        db.write_metadata()
        os.mkdir(db.blocks_dir)
        return db

    @staticmethod
    def load(root_dir):
        print(f"loading database from {root_dir}")
        db = IcosahedronPointDatabase()
        db.root_dir = root_dir
        db.variables_file = os.path.join(root_dir, "variables.txt")
        db.metadata_file = os.path.join(root_dir, "metadata.txt")
        db.blocks_dir = os.path.join(root_dir, "blocks/")
        db.variables_dict = IcosahedronPointDatabase.get_variables_dict_from_file(db.variables_file)
        db.metadata = IcosahedronPointDatabase.get_metadata_from_file(db.metadata_file)
        db.cache = {}
        db.verify_blocks()
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

    def get_block_number_for_point_number(self, pn):
        block_size = self.metadata["block_size"]
        block_number = pn // block_size
        return block_number

    def get_block_fp_for_point_number(self, pn):
        # assume numeric for now, can try ancestry approach later if we are dealing with too many different files and want contiguous points to be more likely to be in the same file
        block_number = self.get_block_number_for_point_number(pn)
        return self.get_block_fp_for_block_number(block_number)

    def get_block_fp_for_block_number(self, block_number):
        block_size = self.metadata["block_size"]
        min_in_block = block_number * block_size
        max_in_block = (block_number + 1) * block_size - 1
        fname = f"Block{block_number}_points_{min_in_block}_to_{max_in_block}.txt"
        return os.path.join(self.blocks_dir, fname)

    def get_all_point_numbers_with_data(self):
        print("getting points with data in database")
        block_numbers = self.get_all_block_numbers_with_data()
        res = set()
        for block_number in block_numbers:
            # print(f"reading block {block_number}")
            pns = self.get_point_numbers_with_data_in_block(block_number)
            assert res & pns == set(), "overlap"
            res |= pns
        print("done getting points with data")
        return res

    def get_point_numbers_with_data_in_block(self, block_number):
        block_size = self.metadata["block_size"]
        block_start = block_number * block_size
        block_fp = self.get_block_fp_for_block_number(block_number)
        with open(block_fp) as f:
            lines = f.readlines()
        res = set()
        for l in lines:
            l = l.strip().split(",")
            adjusted_pn = int(l[0])
            pn = block_start + adjusted_pn
            assert pn not in res
            # print(f"got point number {pn}")
            res.add(pn)
        return res

    def get_all_block_numbers_with_data(self):
        block_files = os.listdir(self.blocks_dir)
        res = set()
        for fname in block_files:
            block_number = int(fname.split("_")[0].replace("Block", ""))
            assert block_number not in res
            res.add(block_number)
        return res

    def get_int_from_line_label(self, line_label):
        N = self.metadata["n_point_code_chars_per_level"]
        assert 1 <= len(line_label) <= N, repr(line_label)
        # base-4 number
        # note that if it is less than N chars long, it has trailing zeros, not leading
        # so 1 acts like 100 and 11 acts like 110 (for N=3)
        to_n = lambda c: "0123".index(c)
        x = 0
        for i, c in enumerate(line_label):
            n = to_n(c)
            x += n * (4 ** ((N-1)-i))
        # print(f"{line_label=} gave {x=}")
        return x

    def get_filepath_and_line_label_for_point_code(self, pc):
        root_dir = self.root_dir
        N = self.metadata["n_point_code_chars_per_level"]
        parent_dir = os.path.join(root_dir, "data_by_point_code/")
        head = pc[0]
        tail = pc[1:]
        assert head in "ABCDEFGHIJKL"
        n_subdirs = (len(tail) - 1) // N
        chars_remaining = len(tail)
        subdirs = []
        for i in range(n_subdirs):
            digits = tail[N*i : N*i+N]
            subdirs.append(digits)
            chars_remaining -= N
        filename = "data.h5"
        line_label = tail[-chars_remaining:]
        fp = os.path.join(parent_dir, head, *subdirs, filename)
        return fp, line_label

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
        self.verify_blocks()

    @staticmethod
    def get_line_from_dict(line_label, d):
        items = [line_label]
        for k,v in sorted(d.items()):
            items.append(f"{k}={v}")
        return ",".join(items)

    def verify_blocks(self):
        # verify that the block files contain the correct point numbers
        block_numbers = self.get_all_block_numbers_with_data()
        failed = False
        for block_number in block_numbers:
            pns = self.get_point_numbers_with_data_in_block(block_number)
            for pn in pns:
                correct_bn = self.get_block_number_for_point_number(pn)
                if correct_bn != block_number:
                    failed = True
                    print(f"found point {pn} in block {block_number} but it belongs in block {correct_bn}")
        if failed:
            raise RuntimeError("points misplaced in blocks")
        else:
            print("blocks verified")


def touch(fp):
    assert not os.path.exists(fp), "cannot touch existing file"
    open(fp, "w").close()


def check_int(value):
    if type(value) is not int:
        raise TypeError(f"Database only accepts int values (you gave {value} of type {type(value)}).\nIf you want enum, make them int shorthands in the condition_array_dir.\nIf you want floats, choose the precision you want and make ints of that, e.g. elevation in millimeters.")


if __name__ == "__main__":
    # trying to see how best to convert the data from point number to point code organization
    # idea to have folders for each group of digits to some size?
    # e.g. C103231 is found in C/102/data.txt, under the line label "231"
    # and C10323102 is found in C/102/231/data.txt, under the line label "02"
    # don't store trailing zeros in point codes here, each point code should be a unique place
    # so 12 top-level directories
    # where should poles go? in their own file maybe? probably best to just do that
    root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"
    db = IcosahedronPointDatabase.load(root_dir)
    pns = db.get_all_point_numbers_with_data()
    # pcs = db.get_all_point_codes_with_data_with_prefix(prefix)
    for pn in pns:
        pc = icm.get_point_code_from_point_number(pn)
        variables = db.get_single_point_all_variables(pn)
        fp, line_label = db.get_filepath_and_line_label_for_point_code(pc)
        line_to_write = IcosahedronPointDatabase.get_line_from_dict(line_label, variables)
        print(pn, pc)
        
        assert fp.endswith("data.h5")
        path = pathlib.Path(fp)

        if path.exists():
            df = pd.read_hdf(fp)
        else:
            parent_path = path.parents[0]  # dir in which data.txt is located
            parent_path.mkdir(parents=True, exist_ok=True)
            var_dict = db.get_variables_dict()
            variable_indices = sorted(var_dict.keys(int, str))
            df = pd.DataFrame(columns=variable_indices)
            # path.touch()
        
        row_index = db.get_int_from_line_label(line_label)
        df.loc[row_index] = variables
        # print(df)
        df.to_hdf(fp, key="df")
    
        # input("check")