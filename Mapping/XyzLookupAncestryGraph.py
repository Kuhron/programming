# data structure to keep track of a point's parents so we can get its coordinates faster
# whenever you add a point, calculate its coordinates and add the parents to this structure
# and recurse, so all the new point's ancestors will be in the structure with their xyz memoized


import random
from functools import reduce
import numpy as np

import IcosahedronMath as icm
import LoadMapData as lmd



# class XyzNode:
#     def __init__(self, point_code, par_node, dpar_node, xyz):
#         self.point_code = point_code
#         self.par_node = par_node
#         self.dpar_node = dpar_node
#         self.xyz = xyz
    
#     def __repr__(self):
#         return f"<{self.point_code} at {self.xyz}>"


class XyzLookupAncestryGraph:
    def __init__(self):
        self.array = []  # according to this, just Python list is pretty good: https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
        self.ln_to_array_index = {}
        self.array_index_to_ln = {}
        self.count = 0
    
    @staticmethod
    def from_list(lns):
        print(f"constructing XyzLookupAncestryGraph from {len(lns)} lookup numbers")
        xyzg = XyzLookupAncestryGraph()
        for i, ln in enumerate(lns):
            if i % 1000 == 0:
                print(f"XyzLookupAncestryGraph.from_list: {i} / {len(lns)}")
            xyzg.add(ln)
        print(f"-- done constructing XyzLookupAncestryGraph")
        return xyzg
    
    def add(self, ln):
        assert type(ln) in [int, np.int64], f"{ln=} of type {type(ln)}"
        if ln in self.ln_to_array_index:
            # print(f"{ln=} already in ancestry graph")
            return
        par_ln = icm.get_parent_from_lookup_number(ln)
        dpar_ln = icm.get_directional_parent_from_lookup_number(ln)
        # print(f"{par_ln=}, {dpar_ln=}")
        assert (par_ln is None and dpar_ln is None) or (par_ln is not None and dpar_ln is not None), "neither or both parents should be None"
        if par_ln is None:
            # original point, base case
            par_index = None
            dpar_index = None
            pn = icm.get_point_number_from_lookup_number(ln)
            xyz = icm.get_xyz_of_initial_point_number(pn)
        else:
            self.add(par_ln)
            self.add(dpar_ln)
            # now that they are in here, get the xyz from the (now-filled-out) parent nodes
            par_index = self.ln_to_array_index[par_ln]
            dpar_index = self.ln_to_array_index[dpar_ln]
            par_xyz = self.array[par_index]
            dpar_xyz = self.array[dpar_index]
            xyz = icm.get_xyz_of_child_from_parent_xyzs(par_xyz, dpar_xyz)
        
        index = self.count
        self.ln_to_array_index[ln] = index
        self.array_index_to_ln[index] = ln
        self.array.append(xyz)
        self.count += 1
    
    def get_xyz(self, ln):
        if ln not in self.ln_to_array_index:
            self.add(ln)
        index = self.ln_to_array_index[ln]
        return self.array[index]
    
    def get_all_xyzs(self):
        res = {ln: self.array[index] for ln, index in self.ln_to_array_index.items()}
        return res
    
    def get_xyzs(self, lns):
        res = {}
        for ln in lns:
            res[ln] = self.get_xyz(ln)  # do this in case we need to add it and its ancestry
        return res

    def get_count(self):
        return self.count
    
    def __getitem__(self, ln):
        return self.get_xyz(ln)



if __name__ == "__main__":
    pc_array = lmd.get_image_pixel_to_icosa_point_code_from_memo("Mienta")
    lns = reduce(lambda x, y: x+y, pc_array)
    lns = random.sample(lns, 46580)
    lns = icm.get_prefix_lookup_numbers_from_point_codes(lns)
    print(len(lns),"points")

    xyzg = XyzLookupAncestryGraph()
    for i, ln in enumerate(lns):
        if i % 1000 == 0:
            print(i, len(lns), xyzg.get_count())
        xyzg.add(ln)
    xyzs = xyzg.get_all_xyzs()
    print("-- done")
    input("check")