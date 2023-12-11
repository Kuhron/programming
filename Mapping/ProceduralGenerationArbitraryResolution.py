# trying to find a way to procedurally generate (i.e., on the fly, lazy evaluation)
# some scalar field such that higher resolution calculation will be deterministic,
# respects the existing points from the lower resolution, and has some kind of scaling property
# like you only get huge mountains on a large scale of the domain, you don't get tiny areas that spike up really far

import random
import string
import numpy as np
import matplotlib.pyplot as plt


# TODO move these binary tree classes to a DataStructures directory in programming dir so I can use it elsewhere
class BinaryTree1D:
    # to help looking at nearby points efficiently
    def __init__(self):
        self.root = None

    def add(self, key, value):
        # print(f"adding {key=}, {value=} to BinaryTree1D")
        # things are organized by key in the tree, and they can carry whatever value they want, the tree doesn't care about that
        if self.root is None:
            # make a node with just this key and some interval around it
            node = BinaryTreeNode(min_key=key-1, center_key=key, max_key=key+1)
            self.root = node
            self.root.add(key, value)
        elif self.root.contains_key(key):
            # if it's equal to the min value, we can't hang it on the center of an interval so we need to make a bigger interval in a new root above this and hang it on that root
            self.root.add(key, value)
        else:
            # make new root and move old root to child of the new one
            key_on_left = self.root.is_off_left_edge(key) or key == self.root.min_key
            if key == self.root.min_key:
                # this can happen if the root goes up to but not including X, then we add key=X
                # then we will make a new root above this where the old root is one of the new children and there is no other child yet
                new_root_max_key = self.root.max_key
                new_root_center_key = self.root.min_key
                distance = new_root_max_key - new_root_center_key
                new_root_min_key = new_root_center_key - distance
                new_node = None
            elif key == self.root.max_key:
                new_root_min_key = self.root.min_key
                new_root_center_key = self.root.max_key
                distance = new_root_center_key - new_root_min_key
                new_root_max_key = new_root_center_key + distance
                new_node = None
            else:
                # make a new node where this key is central and spans the gap between it and the root
                if key_on_left:
                    new_max_key = self.root.min_key
                    new_center_key = key
                    distance = new_max_key - new_center_key
                    new_min_key = key - distance
                else:
                    assert self.root.is_off_right_edge(key)
                    new_min_key = self.root.max_key
                    new_center_key = key
                    distance = new_center_key - new_min_key
                    new_max_key = key + distance

                new_node = BinaryTreeNode(min_key=new_min_key, center_key=new_center_key, max_key=new_max_key)
                # now make a new root with the old root and this new node as children
                new_root_min_key = min(self.root.min_key, new_node.min_key)
                new_root_center_key = self.root.min_key if key_on_left else self.root.max_key
                new_root_max_key = max(self.root.max_key, new_node.max_key)

            new_root = BinaryTreeNode(min_key=new_root_min_key, center_key=new_root_center_key, max_key=new_root_max_key)
            new_root.left_child = new_node if key_on_left else self.root
            new_root.right_child = self.root if key_on_left else new_node
            if new_node is not None:
                new_node.parent = new_root
            self.root.parent = new_root
            self.root = new_root
            self.root.add(key, value)
        # print()
        # print(f"added {key}")
        # print(self)
        # input("check\n")

    def get_values_by_key_range(self, k_min, k_max):
        return self.root.get_values_by_key_range(k_min, k_max)

    @staticmethod
    def test_data_structure():
        def get_test_items_26_letters():
            strs = random.sample(string.ascii_lowercase, 26)
            for c in strs:
                key = string.ascii_lowercase.index(c)
                yield (key, c)

        def get_test_items_floats():
            for i in range(100):
                r = np.random.normal(0, 1)
                yield (r, r)

        for get_test_items in [get_test_items_26_letters, get_test_items_floats]:
            for i in range(10000):
                test_tree = BinaryTree1D()
                for key, value in get_test_items():
                    test_tree.add(key, value)
                print(f"successfully built {i+1} trees", end="\r")
            print()

    def __repr__(self):
        return "BinaryTree1D with root:\n" + repr(self.root)


class BinaryTreeNode:
    # should correspond to the center of some interval, like (1,2) can be broken into two as (1,1.5) and (1.5, 2)
    # values whose keys are exactly equal to the center point should be hung on the root and not descend into the children
    # and so each of those intervals is a node that has a center point and intervals on either side
    # equivalently, can keep track of its min and max key
    # but can be asymmetrical like left=1, center=2, right=7; so allow that too, don't assume equal on each side
    def __init__(self, min_key, center_key, max_key):
        self.min_key = min_key
        self.center_key = center_key
        self.max_key = max_key
        assert self.min_key < self.center_key < self.max_key, f"{min_key=}, {center_key=}, {max_key=}"

        self.left_child = None
        self.right_child = None
        self.parent = None
        self.values = []

    def contains_key(self, key):
        return self.min_key < key and key < self.max_key

    def add(self, key, value):
        # print(f"adding {key=}, {value=} to BinaryTreeNode")
        assert self.contains_key(key)
        if key == self.center_key:
            self.values.append(value)
        elif self.is_on_left(key):
            if self.left_child is None:
                self.left_child = BinaryTreeNode(self.min_key, key, self.center_key)
                self.left_child.parent = self
            self.left_child.add(key, value)
        else:
            if self.right_child is None:
                self.right_child = BinaryTreeNode(self.center_key, key, self.max_key)
                self.right_child.parent = self
            self.right_child.add(key, value)
        # should hang some keys directly from this node if they're equal to center value; otherwise, adding center to the right child causes infinite recursion, or setting new node center not equal to the added key makes it split down a lot to get to float accuracy

    def is_on_left(self, key):
        return self.min_key < key < self.center_key

    def is_on_right(self, key):
        return self.center_key < key < self.max_key

    def is_off_left_edge(self, key):
        return key <= self.min_key

    def is_off_right_edge(self, key):
        return self.max_key <= key

    def has_parent(self):
        return self.parent is not None

    def is_left_child(self):
        return self.parent.left_child is self

    def is_right_child(self):
        return self.parent.right_child is self

    def get_n_levels(self):
        if self.left_child is not None:
            left_levels = 1 + self.left_child.get_n_levels()
        else:
            left_levels = 0
        if self.right_child is not None:
            right_levels = 1 + self.right_child.get_n_levels()
        else:
            right_levels = 0
        return max(left_levels, right_levels)

    def get_values_by_key_range(self, k_min, k_max):
        ?

    def get_print_str_lines(self):
        padding = "| "
        self_str = f"[{self.min_key}, {self.center_key}, {self.max_key}] : {self.values}"
        if self.left_child is not None:
            left_child_strs = self.left_child.get_print_str_lines()
        else:
            left_child_strs = []
        if self.right_child is not None:
            right_child_strs = self.right_child.get_print_str_lines()
        else:
            right_child_strs = []

        left_child_strs = [padding + x for x in left_child_strs]
        right_child_strs = [padding + x for x in right_child_strs]

        strs = left_child_strs + [self_str] + right_child_strs
        return strs

    def __repr__(self):
        lines = self.get_print_str_lines()
        return "\n".join(lines)


class BinaryTree2D:
    # hold (x,y) points in two trees, one for their X coordinate and one for their Y
    def __init__(self):
        self.x_tree = BinaryTree1D()
        self.y_tree = BinaryTree1D()

    def add(self, point):
        # print(f"adding {point} to BinaryTree2D")
        x,y = point
        self.x_tree.add(key=x, value=point)
        self.y_tree.add(key=y, value=point)

    def get_circle_around_point(self, point, radius):
        # first get the square around this point that contains the circle
        # then for each row, start on the left edge, calculate distance, if it's too far then throw it out, repeat inward
        # and do same on right edge
        # can do some more optimizations later like mirror image vertically but this should be fine for now
        square_points = self.get_square_around_point(point, radius)
        print(square_points)
        raise NotImplementedError

    def get_square_around_point(self, point, radius):
        # the radius is from point to each side, so the corners are r*sqrt(2) away
        x,y = point
        points_in_x_range = self.get_points_by_x_range(x - radius, x + radius)
        points_in_y_range = self.get_points_by_y_range(y - radius, y + radius)
        assert type(points_in_x_range) is XYPointSet
        assert type(points_in_y_range) is XYPointSet
        return XYPointSet.intersection(points_in_x_range, points_in_y_range)
        # or could just query on one dimension from the tree and then look through each of those by the other coordinate?

    def get_points_by_x_range(self, x_min, x_max):
        return self.x_tree.get_values_by_key_range(x_min, x_max)

    def get_points_by_y_range(self, y_min, y_max):
        return self.y_tree.get_values_by_key_range(y_min, y_max)

    def __repr__(self):
        return f"PointTree with x_tree:\n{self.x_tree}\n----\nand y_tree:\n{self.y_tree}"


class XYPointSet:
    # for easier iteration over the points based on X and Y coordinates
    def __init__(self):
        self.by_x = {}
        self.by_y = {}

    def add(self, point):
        x,y = point
        same_x = self.by_x.get(x, [])
        self.by_x[x] = sorted(same_x + [point])
        same_y = self.by_y.get(y, [])
        self.by_y[y] = sorted(same_y + [point])



if __name__ == "__main__":
    side_length = 100
    starting_points = [(x,y) for x in range(side_length) for y in range(side_length)]
    values_at_points = {p: 0 for p in starting_points}

    # BinaryTree1D.test_data_structure()

    point_tree = BinaryTree2D()
    for p in random.sample(starting_points, len(starting_points)):
        point_tree.add(p)

    center = (61.3, 17.8)
    radius = 4.4
    pts = point_tree.get_circle_around_point(center, radius)
    print(pts)

