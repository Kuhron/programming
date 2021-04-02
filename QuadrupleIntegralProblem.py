# the math problem I made up in late high school to be a waste of time on purpose
# integral(0,1) integral(0,1) integral(0,1) integral(0,1) det(A) dw dx dy dz
# where A is the product of all 2x2 matrices permuting [[w,x],[y,z]] (so there are 4! = 24 of them), and if order matters, they are in alphabetical order where you read left to right then top to bottom

import sympy
from sympy.abc import w,x,y,z
from sympy import poly, integrate
import itertools


def get_variable_permutations_in_order():
    perms = itertools.permutations("wxyz")
    d = {"w":w, "x":x, "y":y, "z":z}
    def convert_str(perm):
        return [d[char] for char in perm]
    return [convert_str(perm) for perm in sorted(perms)]


def get_2x2_det(permutation):
    a,b,c,d = permutation  # it will be some order of w,x,y,z
    return poly(a*d-b*c)


def get_all_dets_product():
    perms = get_variable_permutations_in_order()
    p = 1
    for perm in perms:
        p *= get_2x2_det(perm)
    return p



if __name__ == "__main__":
    perms = get_variable_permutations_in_order()
    p = get_all_dets_product()
    p_expr = p.as_expr()
    integral = integrate(integrate(integrate(integrate(p_expr, (z,0,1)), (y,0,1)), (x,0,1)), (w,0,1))
    print(integral)
    # it says the answer is 10189641213114751/333192503101628570400
    # don't think I believe that, very odd
