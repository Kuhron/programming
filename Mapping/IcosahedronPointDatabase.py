# store data indexed by icosahedral lattice point number
# multiple csvs by some block size e.g. 2**14
# all of them need the same header
# but need to allow sparseness, e.g. many points won't have values for certain variables, e.g. suppose "salt flat" is undefined most places but has values in small regions of the world, don't want to waste space having that defined as zero for almost every point
# maybe have a format that indexes the variables, every time a new variable is added it gets the next index number
# there is a global file telling which index means which variable, e.g. {0:is_land, 1:elevation, 2:volcanism, etc.}
# then the rows in the files look something like: 40128,0=0,1=-65.3,5=17.1,24=0
# so they're only showing the variables they're specified for


import os
from BiDict import BiDict


class IcosahedronPointDatabase:
    def __init__(self):
        pass

    @staticmethod
    def new(root_dir, block_size):
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
        }
        db.cache = {}
        touch(db.variables_file)
        db.write_metadata()
        os.mkdir(db.blocks_dir)
        return db

    @staticmethod
    def load(root_dir):
        db = IcosahedronPointDatabase()
        db.root_dir = root_dir
        db.variables_file = os.path.join(root_dir, "variables.txt")
        db.metadata_file = os.path.join(root_dir, "metadata.txt")
        db.blocks_dir = os.path.join(root_dir, "blocks/")
        db.variables_dict = IcosahedronPointDatabase.get_variables_dict_from_file(db.variables_file)
        db.metadata = IcosahedronPointDatabase.get_metadata_from_file(db.metadata_file)
        db.cache = {}
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

    def get(self, point_number, variable_name):
        if point_number in self.cache:
            return self.cache[point_number][variable_name]
        else:
            return self.get_from_file(point_number, variable_name)

    def get_from_file(self, point_number, variable_name):
        block_fp = self.get_block_fp(point_number)
        if not os.path.exists(block_fp):
            # raise KeyError(f"no data for point {point_number}")
            return None
        with open(block_fp) as f:
            lines = f.readlines()

        # find the line starting with this point, if any
        block_size = self.metadata["block_size"]
        block_number = point_number // block_size
        block_start = block_number * block_size
        adjusted_point_number = point_number - block_start
        assert 0 <= adjusted_point_number < block_size, (point_number, block_size, block_number, block_start, adjusted_point_number)
        lines = [l.strip().split(",") for l in lines]
        lines_this_point = [l for l in lines if int(l[0]) == adjusted_point_number]
        if len(lines_this_point) == 0:
            # raise KeyError(f"no data for point {point_number}")
            return None
        elif len(lines_this_point) > 1:
            raise RuntimeError(f"point {point_number} found more than once!")
        else:
            l, = lines_this_point
            variable_number = self.get_variable_number_from_name(variable_name)
            # the first index (0) in l is the point number
            d = {}
            for x in l[1:]:
                k,v = x.split("=")
                k = int(k)
                v = int(v)
                d[k] = v
            # typ = self.get_variable_type_from_name(variable_name)
            # for now, make everything in the db an int, can capture enums, bools, and floats to some precision, and that way I don't have to parse a file to figure out what the types are supposed to be; put units in the varname if you care about that, e.g. elevation_meters
            val = d[variable_number]
            self.add_to_cache(point_number, variable_name, val)
            return val

    def add_to_cache(self, point_number, variable_name, value):
        if point_number not in self.cache:
            self.cache[point_number] = {variable_name: value}
        else:
            self.cache[point_number][variable_name] = value

    def set(self, point_number, variable_name, value, write=False):
        variable_number = self.get_variable_number_from_name(variable_name)
        self.add_to_cache(point_number, variable_name, value)
        if write:
            self.write()  # don't do this too often or it will be slow

    def get_block_fp(self, point_number):
        # assume numeric for now, can try ancestry approach later if we are dealing with too many different files and want contiguous points to be more likely to be in the same file
        block_size = self.metadata["block_size"]
        block_number = point_number // block_size
        return self.get_block_fp_for_block_number(block_number)

    def get_block_fp_for_block_number(self, block_number):
        block_size = self.metadata["block_size"]
        min_in_block = block_number * block_size
        max_in_block = (block_number + 1) * block_size - 1
        fname = f"Block{block_number}_points_{min_in_block}_to_{max_in_block}.txt"
        return os.path.join(self.blocks_dir, fname)

    def get_all_point_numbers_with_data(self):
        block_numbers = self.get_all_block_numbers_with_data()
        res = set()
        for block_number in block_numbers:
            point_numbers = self.get_point_numbers_with_data_in_block(block_number)
            assert res & point_numbers == set(), "overlap"
            res |= point_numbers
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

    def get_variable_number_from_name(self, name):
        return self.variables_dict[name]

    def get_variable_name_from_number(self, number):
        return self.variables_dict[number]

    def __getitem__(self, tup):
        point_number, variable_name = tup
        return self.get(point_number, variable_name)

    def __setitem__(self, tup, val):
        point_number, variable_name = tup
        self.set(point_number, variable_name, val)

    def clear_cache(self):
        self.cache = {}
        print("cache cleared")

    def write(self, clear_cache=True):
        # update the block files on disk
        block_size = self.metadata["block_size"]
        blocks_in_cache = set(point_number // block_size for point_number in self.cache.keys())
        for block_number in blocks_in_cache:
            print("block_number", block_number)
            block_start = block_number * block_size
            point_numbers = set([p for p in self.cache.keys() if p // block_size == block_number])
            fps = set(self.get_block_fp(p) for p in point_numbers)
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
            all_points_to_write = sorted(set(data_on_disk.keys()) | set(point_numbers))
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
                adjusted_point_number = p - block_start
                assert 0 <= adjusted_point_number < block_size, (p, block_size, block_number, block_start, adjusted_point_number)
                new_l = IcosahedronPointDatabase.line_from_dict(adjusted_point_number, d)
                lines_to_write.append(new_l)
            with open(fp, "w") as f:
                for l in lines_to_write:
                    f.write(l + "\n")
        print("db written")
        if clear_cache:
            self.clear_cache()

    @staticmethod
    def line_from_dict(point_number, d):
        items = [str(point_number)]
        for k,v in sorted(d.items()):
            items.append(f"{k}={v}")
        return ",".join(items)


def touch(fp):
    assert not os.path.exists(fp), "cannot touch existing file"
    open(fp, "w").close()

