# box-corner mapping is where you take an original half-peel
# e.g. the square (two triangle faces) with C at the upper right
# C at upper right has upper left of A, lower left of K, lower right of L
# D at upper right has upper left of C, lower left of L, lower right of B
# other half-peel boxes are the same, just add the peel-analogue offset (C,D=E,F=G,H=I,J=K,L)
# see what happens to the corners of the boxes (which are dpars of parts of the path)
# if we replace the upper right corner (ancestor) with one of its children
# e.g. we want to compare the paths to get to C2131 vs C131,
# which is a C vs C2 mapping for the first step
# C -> C2 makes the corners A -> K1, K -> K (=K0), and L -> L1
# these mappings exist for all four steps you can take from an ancestor (0, 1, 2, and 3)
# corresponding to the quadrant of the half-peel box where that step is the upper right
# so for C -> C0 you just shrink to the upper right box quadrant
# and that gives you A -> C1, K -> C2, L -> C3
# etc.


PEELS = ["CD", "EF", "GH", "IJ", "KL"]


def get_directional_parent_from_point_code_using_box_corner_mapping(point_code, mapping_stack=None):
    if mapping_stack is None:
        mapping_stack = []
    # assumes the peel is normalized to CD
    mappings = {
        # need A0 and B0 for when we're dealing with lengthening of reverse-polarity codes
        # similarly with using A3 instead of K1 and B1 instead of L3
        "C0": [("C","C0"), ("A","C1"), ("K","C2"), ("L","C3")],
        "C1": [("C","C1"), ("A","A0"), ("K","A3"), ("L","C2")],
        "C2": [("C","C2"), ("A","A3"), ("K","K0"), ("L","L1")],
        "C3": [("C","C3"), ("A","C2"), ("K","L1"), ("L","L0")],
        "D0": [("D","D0"), ("C","D1"), ("L","D2"), ("B","D3")],
        "D1": [("D","D1"), ("C","C0"), ("L","C3"), ("B","D2")],
        "D2": [("D","D2"), ("C","C3"), ("L","L0"), ("B","B1")],
        "D3": [("D","D3"), ("C","D2"), ("L","B1"), ("B","B0")],
    }
    base_case_dpars = {
        "C1":"A", "C2":"K", "C3":"L",
        "D1":"C", "D2":"L", "D3":"B",
    }

    try:
        dpar = base_case_dpars[point_code]
        # print(f"known type: iteration 1, giving dpar {dpar} of point {point_code}")
        return dpar
    except KeyError:
        pass
    
    assert len(point_code) >= 3, f"shouldn't use box corner mapping with point from iteration 0 or 1, but got {point_code}"
    new_point_code, mapping_stack = shorten_by_box_corner_mapping(point_code, mapping_stack, mappings)
    new_dpar = get_directional_parent_from_point_code_using_box_corner_mapping(new_point_code, mapping_stack)
    # print(f"got new_dpar {new_dpar}")

    if dpar_is_on_reversed_edge_from_perspective_of_point(new_dpar, point_code):
        # print(f"dpar {new_dpar} is on reversed edge (but not in reverse-polarity encoding")
        new_dpar = reverse_edge_polarity(new_dpar)
        # print(f"dpar reverse-encoded to {new_dpar}")
    
    dpar, mapping_stack = lengthen_by_box_corner_mapping(new_dpar, mapping_stack, mappings)
    
    # try leaving it in reverse-polarity-encoding for recursive calls
    # then clean it up by correcting polarity encoding once have answer
    return dpar


def shorten_by_box_corner_mapping(point_code, mapping_stack, mappings):
    prefix = point_code[:2]
    tail = point_code[2:]
    mapping = mappings[prefix]

    # apply the mapping to the point
    # the shorter ones are first in the mapping tuples
    new_prefix = None
    for shorter, longer in mapping:
        if longer == prefix:
            new_prefix = shorter
            break
    assert new_prefix is not None, f"failed to get new prefix for shortening code {point_code} by mapping {mapping}"
    new_point_code = new_prefix + tail

    # add the mapping to the stack (using the prefix as shorthand)
    mapping_stack = mapping_stack + [prefix]
    # print(f"shortened {point_code} to {new_point_code} using mapping {mapping}")
    return new_point_code, mapping_stack


def lengthen_by_box_corner_mapping(point_code, mapping_stack, mappings):
    # use the last mapping on the stack
    prefix = mapping_stack[-1]
    # remove the mapping from the stack so it's not reused
    mapping_stack = mapping_stack[:-1]
    mapping = mappings[prefix]

    point_prefix = point_code[0]
    tail = point_code[1:]
    new_prefix = None
    for shorter, longer in mapping:
        if shorter == point_prefix:
            new_prefix = longer
            break
    assert new_prefix is not None, f"failed to get new prefix for lengthening code {point_code} by mapping {mapping}"
    new_point_code = new_prefix + tail
    # print(f"lengthened {point_code} to {new_point_code} using mapping {mapping}")
    return new_point_code, mapping_stack


def dpar_is_on_reversed_edge_from_perspective_of_point(dpar, reference_point_code):
    # get peel of reference point, it should be normalized to C-D peel
    orig = reference_point_code[0]
    if orig not in ["C", "D"]:
        raise ValueError(f"please normalize peel for point {reference_point_code} to C-D")

    # e.g. from the perspective of the C-D peel, the K-A edge runs the wrong way
    # you'd want it to run down (toward K) from A (so in the 3 direction from A)
    # but instead it runs up (toward A) from K (so in the 1 direction from K)
    # similarly the L-B edge runs to the right, in the 3 direction from L
    # but we want it to run to the left, in the 1 direction from B
    # so we will code points on these *as if* they do this,
    # for the purposes of box corner mapping to find dpars
    # and then convert the result back to the correct notation

    # don't treat K/L as being reversed, just the points on the edge itself
    if len(dpar) == 1:
        return False

    tail = dpar[1:]
    uses_ones = all(x in ["0", "1"] for x in tail)
    uses_threes = all(x in ["0", "3"] for x in tail)
    has_non_zero = any(x != "0" for x in tail)
    is_on_k_a_edge = dpar[0] == "K" and uses_ones and has_non_zero
    is_on_l_b_edge = dpar[0] == "L" and uses_threes and has_non_zero
    return is_on_k_a_edge or is_on_l_b_edge


def flip_prefix_for_edge_reversal(x, reference_peel="KL"):
    assert reference_peel in PEELS
    assert len(x) == 1, x
    k, l = reference_peel
    return {"A":k, k:"A", "B":l, l:"B"}[x]


def flip_tail_for_edge_reversal(tail):
    # convert the non-zeros to ones and treat as a binary decimal, then subtract from 1
    if all(x in ["0", "1"] for x in tail):
        tail_has_ones_or_threes = "1"
    elif all(x in ["0", "3"] for x in tail):
        tail_has_ones_or_threes = "3"
        tail = tail.replace("3", "1")
    else:
        raise Exception("shouldn't happen, tail is " + tail)
    
    assert all(x in ["0", "1"] for x in tail), tail
    # just subtract from binary the way I know how to subtract from power of 10
    # where every pair adds up to the base-1 except the last one, which adds to the base
    new_tail = ""
    dig = "3" if tail_has_ones_or_threes == "1" else "1" if tail_has_ones_or_threes == "3" else None
    for x in tail[:-1]:
        if x == "0":
            # flip the ones and threes
            new_tail += dig
        elif x == "1":
            new_tail += "0"
        else:
            raise Exception("shouldn't happen")
    
    # now do the last one
    assert tail[-1] == "1", f"tail can't have trailing zeros: {tail}"
    new_tail += dig
    
    return new_tail


def reverse_edge_polarity(point_code, reference_peel="KL"):
    # points of form K{0,1}+ can be reverse-polarity-coded as A{0,3}+
    # similarly L{0,3}+ can be reverse-polarity-coded as B{0,1}+
    # (this is because of the reversed edges messing up dpar finding)
    # this function allows it to go either way
    assert reference_peel in PEELS
    prefix = point_code[0]
    tail = point_code[1:]
    tail_no_trailing_zeros, trailing_zeros = separate_trailing_zeros(tail)
    new_prefix = flip_prefix_for_edge_reversal(prefix, reference_peel)
    new_tail_no_trailing_zeros = flip_tail_for_edge_reversal(tail_no_trailing_zeros)
    new_tail = new_tail_no_trailing_zeros + trailing_zeros
    new_point_code = new_prefix + new_tail
    assert len(new_point_code) == len(point_code), f"got wrong length code {new_point_code} from reversing {point_code}"
    return new_point_code


def correct_reversed_edge_polarity(point_code, reference_peel):
    # this function only wants codes that are already in reverse-polarity-coded form
    if not point_code_is_in_reversed_polarity_encoding(point_code):
        # it's not in improper-polarity mode
        raise ValueError(f"code {point_code} is not in reverse-polarity-coded form")
    return reverse_edge_polarity(point_code, reference_peel)


def point_code_is_in_reversed_polarity_encoding(point_code):
    prefix = point_code[0]
    tail = point_code[1:]
    uses_ones = all(x in ["0", "1"] for x in tail)
    uses_threes = all(x in ["0", "3"] for x in tail)
    has_non_zero = any(x != "0" for x in tail)
    is_on_k_a_edge = prefix == "A" and uses_threes and has_non_zero
    is_on_l_b_edge = prefix == "B" and uses_ones and has_non_zero
    return is_on_k_a_edge or is_on_l_b_edge


def separate_trailing_zeros(s):
    zeros = ""
    while s[-1] == "0":
        zeros += "0"
        s = s[:-1]
    return s, zeros


def get_peel_containing_point_code(pc):
    res, = [peel for peel in PEELS if pc[0] in peel]  # should be exactly one of CD, EF, GH, IJ, KL
    return res
