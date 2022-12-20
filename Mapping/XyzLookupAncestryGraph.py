# data structure to keep track of a point's parents so we can get its coordinates faster
# whenever you add a point, calculate its coordinates and add the parents to this structure
# and recurse, so all the new point's ancestors will be in the structure with their xyz memoized


import random
from functools import reduce

import IcosahedronMath as icm
import LoadMapData as lmd



class XyzNode:
    def __init__(self, point_code, par_node, dpar_node, xyz):
        self.point_code = point_code
        self.par_node = par_node
        self.dpar_node = dpar_node
        self.xyz = xyz
    
    # def get_parent_point_code(self):
    #     return icm.get_parent_from_point_code(self.point_code)
    
    # def get_directional_parent_point_code(self):
    #     return icm.get_directional_parent_from_point_code(self.point_code)
    
    def __repr__(self):
        return f"<{self.point_code} at {self.xyz}>"


class XyzLookupAncestryGraph:
    def __init__(self):
        self.pc_to_node = {}
    
    @staticmethod
    def from_list(pcs):
        xyzg = XyzLookupAncestryGraph()
        for pc in pcs:
            xyzg.add(pc)
        return xyzg
    
    def add(self, pc):
        if pc in self.pc_to_node:
            # print(f"{pc=} already in ancestry graph")
            return
        par_pc = icm.get_parent_from_point_code(pc)
        dpar_pc = icm.get_directional_parent_from_point_code(pc)
        assert (par_pc is None and dpar_pc is None) or (par_pc is not None and dpar_pc is not None)
        if par_pc is None:
            # original point, base case
            par_node = None
            dpar_node = None
            xyz = icm.get_xyz_from_point_code(pc)
        else:
            self.add(par_pc)
            self.add(dpar_pc)
            # now that they are in here, get the xyz from the (now-filled-out) parent nodes
            par_node = self.pc_to_node[par_pc]
            dpar_node = self.pc_to_node[dpar_pc]
            par_xyz = par_node.xyz
            dpar_xyz = dpar_node.xyz
            xyz = icm.get_xyz_of_child_from_parent_xyzs(par_xyz, dpar_xyz)
        
        node = XyzNode(pc, par_node, dpar_node, xyz)
        self.pc_to_node[pc] = node
    
    def get_xyz(self, pc):
        if pc not in self.pc_to_node:
            self.add(pc)
        node = self.pc_to_node[pc]
        return node.xyz
    
    def get_all_xyzs(self):
        res = {pc: node.xyz for pc, node in self.pc_to_node.items()}
        return res
    
    def get_xyzs(self, pcs):
        res = {}
        for pc in pcs:
            res[pc] = self.get_xyz(pc)  # do this in case we need to add it and its ancestry
        return res

    def get_count(self):
        return len(self.pc_to_node)



if __name__ == "__main__":
    pc_array = lmd.get_image_pixel_to_icosa_point_code_from_memo("Mienta")
    pcs = reduce(lambda x, y: x+y, pc_array)
    pcs = random.sample(pcs, 10000)
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