# data structure to tell which point codes are in a large set

import IcosahedronMath as icm


class PointCodeTrie:
    def __init__(self):
        self.dct = {}
        self.count = 0
    
    @staticmethod
    def from_list(lst):
        trie = PointCodeTrie()
        for x in lst:
            trie.add(x)
        return trie

    def add(self, s):
        assert any(s.startswith(x) for x in "ABCDEFGHIJKL"), s
        assert all(x in list("0123") for x in s[1:]), s
        assert s[-1] != "0", s
        # print(f"adding {s} to trie")
        d = self.dct
        for c in s:
            # print(f"c is now {c}")
            if c not in d:
                d[c] = {}
            d = d[c]
            # print(f"d is now {d}")
        if None in d:
            # this endpoint already exists
            # print(f"string {s} was already added")
            pass
            # don't update the count
        else:
            d[None] = None
            self.count += 1
    
    def contains(self, s):
        d = self.dct
        for c in s:
            if c not in d:
                return False
            d = d[c]
        return None in d
        # otherwise the endpoint isn't there
        # it's just a substring of something in the dict

    def remove(self, s):
        d = self.dct
        for c in s:
            if c not in d:
                raise ValueError(f"{s} not in trie")
            d = d[c]
        if None not in d:
            raise ValueError(f"{s} not in trie")
        d.remove(None)
        self.count -= 1
    
    def get_all(self):
        # walk in order
        return PointCodeTrie.get_all_strings_in_trie_dict(self.dct)

    @staticmethod
    def get_all_strings_in_trie_dict(d, prefix=""):
        res = []
        for k in d.keys():
            if k is None:
                assert d[k] is None
                res.append("")  # endpoint reached
            else:
                sub_d = d[k]
                sub_prefix = k
                res += PointCodeTrie.get_all_strings_in_trie_dict(sub_d, sub_prefix)
        res = sorted(res)
        return [prefix + s for s in res]



if __name__ == "__main__":
    trie = PointCodeTrie()
    set1 = set()

    # trie.add("C123123")
    # print(trie.contains("C123"))  # should be False
    # print(trie.contains("C123123123"))  # should be False

    while trie.count < 100:
        pc = icm.get_random_point_code(min_iterations=0, expected_iterations=5, max_iterations=12)
        trie.add(pc)
        set1.add(pc)
        # print(f"current count: {trie.count}")

    print("done adding, now checking")
    set2 = set()
    for pc in trie.get_all():
        print(f"got pc from trie: {pc}")
        set2.add(pc)
    s1not2 = set1 - set2
    s2not1 = set2 - set1
    print("s1not2:", sorted(s1not2))
    print("s2not1:", sorted(s2not1))
    assert set1 == set2
