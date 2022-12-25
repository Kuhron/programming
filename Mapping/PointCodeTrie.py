# data structure to tell which point codes are in a large set

import random

import IcosahedronMath as icm
import IcosahedronPointDatabase as icdb


class PointCodeTrie:
    def __init__(self):
        self.dct = {}
        self.count = 0
    
    @staticmethod
    def from_list(lst):
        trie = PointCodeTrie()
        for x in lst:
            trie.add_point_code(x)
        return trie

    def add_point_code(self, pc):
        pv = icm.get_place_value_array_from_point_code(pc)
        self.add_place_value_array(pv)

    def add_place_value_array(self, pv):
        d = self.dct
        for n in pv:
            if n not in d:
                d[n] = {}
            d = d[n]
        if -1 in d:
            # this endpoint already exists
            # don't update the count
            pass
        else:
            d[-1] = -1
            self.count += 1
    
    def contains_point_code(self, pc):
        pv = icm.get_place_value_array_from_point_code(pc)
        return self.contains_place_value_array(pv)
    
    def contains_place_value_array(self, pv):
        d = self.dct
        for n in pv:
            if n not in d:
                return False
            d = d[n]
        return -1 in d
        # otherwise the endpoint isn't there
        # it's just a substring of something in the dict

    def remove_point_code(self, pc):
        pv = icm.get_place_value_array_from_point_code(pc)
        self.remove_place_value_array(pv)
    
    def remove_place_value_array(self, pv):
        d = self.dct
        for n in pv:
            if n not in d:
                raise ValueError(f"{pv} not in trie")
            d = d[n]
        if -1 not in d:
            raise ValueError(f"{pv} not in trie")
        d.remove(-1)
        self.count -= 1
    
    def get_all(self):
        # walk in order
        return PointCodeTrie.get_all_strings_in_trie_dict(self.dct)

    @staticmethod
    def get_all_strings_in_trie_dict(d):
        pvs = PointCodeTrie.get_all_place_value_arrays_in_trie_dict(d)
        res = [icm.get_point_code_from_place_value_array(pv) for pv in pvs]
        return res

    @staticmethod
    def get_all_place_value_arrays_in_trie_dict(d, prefix=None):
        if prefix is None:
            prefix = []  # no mutable defaults
        res = []
        for k in d.keys():
            if k == -1:
                assert d[k] == -1
                res.append([])  # endpoint reached
            else:
                sub_d = d[k]
                sub_prefix = [k]
                res += PointCodeTrie.get_all_place_value_arrays_in_trie_dict(sub_d, sub_prefix)
        res = sorted(res)
        return [prefix + pv for pv in res]


def test_trie():
    trie = PointCodeTrie()
    set1 = set()
    while trie.count < 100:
        pc = icm.get_random_point_code(min_iterations=0, expected_iterations=5, max_iterations=12)
        trie.add_point_code(pc)
        set1.add(pc)
        # print(f"current count: {trie.count}")

    print("done adding, now checking")
    set2 = set()
    for pc in trie.get_all():
        print(f"got pc from trie: {pc}")
        set2.add(pc)
    s1not2 = set1 - set2
    s2not1 = set2 - set1
    # print("s1not2:", sorted(s1not2))
    # print("s2not1:", sorted(s2not1))
    assert set1 == set2



if __name__ == "__main__":
    test_trie()
    db_root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"
    db = icdb.IcosahedronPointDatabase.load(db_root_dir)
    df = db.df
    region_radius_gc = 0.03
    region_center_ln = random.choice(df.index)
    region_center_pc = icm.get_point_code_from_prefix_lookup_number(region_center_ln)
    pcs_with_data_in_region = icdb.get_lookup_numbers_in_database_in_region(db, region_center_pc, region_radius_gc, use_narrowing=True, lns_to_consider=None)
    trie = PointCodeTrie.from_list(pcs_with_data_in_region)
    print(f"{trie.count=}")
    