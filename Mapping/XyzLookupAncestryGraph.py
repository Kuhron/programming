# data structure to keep track of a point's parents so we can get its coordinates faster
# whenever you add a point, calculate its coordinates and add the parents to this structure
# and recurse, so all the new point's ancestors will be in the structure with their xyz memoized


import random
from functools import reduce

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
        self.pc_to_array_index = {}
        self.array_index_to_pc = {}
        self.count = 0
    
    @staticmethod
    def from_list(pcs):
        print(f"constructing XyzLookupAncestryGraph from {len(pcs)} point codes")
        xyzg = XyzLookupAncestryGraph()
        for i, pc in enumerate(pcs):
            if i % 1000 == 0:
                print(f"{i} / {len(pcs)}")
            xyzg.add(pc)
        print(f"done constructing XyzLookupAncestryGraph")
        return xyzg
    
    def add(self, pc):
        if pc in self.pc_to_array_index:
            # print(f"{pc=} already in ancestry graph")
            return
        par_pc = icm.get_parent_from_point_code(pc)
        dpar_pc = icm.get_directional_parent_from_point_code(pc)
        assert (par_pc is None and dpar_pc is None) or (par_pc is not None and dpar_pc is not None), "neither or both parents should be None"
        if par_pc is None:
            # original point, base case
            par_index = None
            dpar_index = None
            xyz = icm.get_xyz_from_point_code(pc)
        else:
            self.add(par_pc)
            self.add(dpar_pc)
            # now that they are in here, get the xyz from the (now-filled-out) parent nodes
            par_index = self.pc_to_array_index[par_pc]
            dpar_index = self.pc_to_array_index[dpar_pc]
            par_xyz = self.array[par_index]
            dpar_xyz = self.array[dpar_index]
            xyz = icm.get_xyz_of_child_from_parent_xyzs(par_xyz, dpar_xyz)
        
        index = self.count
        self.pc_to_array_index[pc] = index
        self.array_index_to_pc[index] = pc
        self.array.append(xyz)
        self.count += 1
    
    def get_xyz(self, pc):
        if pc not in self.pc_to_array_index:
            self.add(pc)
        index = self.pc_to_array_index[pc]
        return self.array[index]
    
    def get_all_xyzs(self):
        res = {pc: self.array[index] for pc, index in self.pc_to_array_index.items()}
        return res
    
    def get_xyzs(self, pcs):
        res = {}
        for pc in pcs:
            res[pc] = self.get_xyz(pc)  # do this in case we need to add it and its ancestry
        return res

    def get_count(self):
        return self.count
    
    def __getitem__(self, pc):
        return self.get_xyz(pc)



if __name__ == "__main__":
    pc_array = lmd.get_image_pixel_to_icosa_point_code_from_memo("Mienta")
    pcs = reduce(lambda x, y: x+y, pc_array)
    pcs = random.sample(pcs, 46580)
    print(len(pcs),"points")

    xyzg = XyzLookupAncestryGraph()
    for i, pc in enumerate(pcs):
        if i % 1000 == 0:
            print(i, len(pcs), xyzg.get_count())
        xyzg.add(pc)
    # for pc, node in xyzg.pc_to_node.items():
    #     print(node)
    xyzs = xyzg.get_all_xyzs()
    print("done")
    input("check")