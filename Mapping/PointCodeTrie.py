# data structure to tell which point codes are in a large set

import random

import IcosahedronMath as icm
import IcosahedronPointDatabase as icdb


class PointCodeTrie:
    LETTER_TO_NUMBER_DICT = {c:i for i,c in enumerate("CDEFGHIJKL")}
    LETTER_TO_NUMBER_DICT["A"] = -2
    LETTER_TO_NUMBER_DICT["B"] = -3
    NUMBER_TO_LETTER_DICT = {i:c for c,i in LETTER_TO_NUMBER_DICT.items()}

    def __init__(self):
        self.dct = {}
        self.count = 0
    
    @staticmethod
    def from_list(lst):
        trie = PointCodeTrie()
        for x in lst:
            trie.add_point_code(x)
        return trie
    
    @staticmethod
    def point_code_to_number_array(pc):
        assert pc[-1] != "0", pc
        head = pc[0]
        tail = pc[1:]
        n0 = PointCodeTrie.LETTER_TO_NUMBER_DICT[head]
        res = [n0] + [int(x) for x in tail]
        return res
    
    @staticmethod
    def number_array_to_point_code(nums):
        head = nums[0]
        tail = nums[1:]
        c0 = PointCodeTrie.NUMBER_TO_LETTER_DICT[head]
        res = c0 + "".join(str(n) for n in tail)
        return res

    def add_point_code(self, pc):
        nums = PointCodeTrie.point_code_to_number_array(pc)
        self.add_number_array(nums)

    def add_number_array(self, nums):
        d = self.dct
        for n in nums:
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
        nums = PointCodeTrie.point_code_to_number_array(pc)
        return self.contains_number_array(nums)
    
    def contains_number_array(self, nums):
        d = self.dct
        for n in nums:
            if n not in d:
                return False
            d = d[n]
        return -1 in d
        # otherwise the endpoint isn't there
        # it's just a substring of something in the dict

    def remove_point_code(self, pc):
        nums = PointCodeTrie.point_code_to_number_array(pc)
        self.remove_number_array(nums)
    
    def remove_number_array(self, nums):
        d = self.dct
        for n in nums:
            if n not in d:
                raise ValueError(f"{nums} not in trie")
            d = d[n]
        if -1 not in d:
            raise ValueError(f"{nums} not in trie")
        d.remove(-1)
        self.count -= 1
    
    def get_all(self):
        # walk in order
        return PointCodeTrie.get_all_strings_in_trie_dict(self.dct)

    @staticmethod
    def get_all_strings_in_trie_dict(d):
        number_arrays = PointCodeTrie.get_all_number_arrays_in_trie_dict(d)
        res = [PointCodeTrie.number_array_to_point_code(nums) for nums in number_arrays]
        return res

    @staticmethod
    def get_all_number_arrays_in_trie_dict(d, prefix=None):
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
                res += PointCodeTrie.get_all_number_arrays_in_trie_dict(sub_d, sub_prefix)
        res = sorted(res)
        return [prefix + nums for nums in res]


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
    region_center_pc = random.choice(df.index)
    pcs_with_data_in_region = icdb.get_point_codes_in_database_in_region(db, region_center_pc, region_radius_gc, use_narrowing=True, pcs_to_consider=None)
    trie = PointCodeTrie.from_list(pcs_with_data_in_region)
    print(f"{trie.count=}")
    input("a")