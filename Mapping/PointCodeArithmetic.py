# implementing the "arithmetic" on point code strings
# to try to remove the need for recursion as much as possible in computing adjacency

from BoxCornerMapping import correct_reversed_edge_polarity, point_code_is_in_reversed_polarity_encoding, reverse_edge_polarity


NORTHERN_RING = ["C", "E", "G", "I", "K"]
SOUTHERN_RING = ["D", "F", "H", "J", "L"]


def add_direction_to_point_code(pc, x, fix_edge_polarity=True):
    if pc in [None, "?"]:
        return pc

    pc, peel_offset = normalize_peel(pc)
    reference_peel = "CD"  # since we normalized, but need to make sure to update this if we move elsewhere
    head = pc[0]
    tail = pc[1:]
    initial_points_results = {
        "C": {1: "A", 2: "K", 3: "L", -1: "D", -2: "E", -3: None},
        "D": {1: "C", 2: "L", 3: "B", -1: None, -2: "F", -3: "E"},
        "A": {3: "K"},  # for reversed K-A edge
        "B": {1: "L"},  # for reversed L-B edge
    }

    # print(f"adding {x:+} to {pc}")

    if fix_edge_polarity and pc in ["A", "B"]:
        # fix_edge_polarity indicates that we are in top-level call
        # it's okay to do like A + 3 = K in intermediate calls (with reversed K-A edge)
        raise ValueError(f"directions from poles are ill-defined: {pc=}, {x=}")
    elif len(pc) == 1:
        res = initial_points_results[pc][x]
        # print(f"case: initial point; {res=}")
    else:
        on_edge_CA = head == "C" and all(y in ["0", "1"] for y in tail)
        on_edge_DB = head == "D" and all(y in ["0", "3"] for y in tail)

        verify_valid_direction(pc, x)

        if x == 2:
            y1 = add1(pc)
            y13 = add3(y1)
            y3 = add3(pc)
            y31 = add1(y3)
            # print(f"{pc} +1+3 = {y13} ; {pc} +3+1 = {y31}")
            assert y13 == y31
            res = y13
            if res[0] in ["A", "B"]:
                # we moved off the left or bottom edge of CD peel
                reference_peel = "KL"
            # print(f"case: x=2; {res=}")
        
        # special case of C1 - 2 = E2, calculating this as -1-3 gives None
        # similarly C1 - 3 = E1; D3 - 2 = F2; D3 - 1 = F3
        elif pc == "C1" and x == -2:
            res = "E2"
            # print(f"case: C1 - 2; {res=}")
        elif pc == "C1" and x == -3:
            res = "E1"
            # print(f"case: C1 - 3; {res=}")
        elif pc == "D3" and x == -2:
            res = "F2"
            # print(f"case: D3 - 2; {res=}")
        elif pc == "D3" and x == -1:
            res = "F3"
            # print(f"case: D3 - 1; {res=}")

        # refraction cases
        elif on_edge_CA and x == -3:
            if all(y == "0" for y in tail):
                # we will get extraneous solution where p-3 = p-2, but p-3 should be undefined
                res = None
                # print(f"case: on edge CA, x=-3, tail all zero; {res=}")
            else:
                new_head = "E"
                new_tail = replace_zeros_with_twos(tail)
                res = new_head + new_tail
                # print(f"case: on edge CA, x=-3, tail not all zero; {res=}")
        elif on_edge_CA and x == -2:
            # do the refracting last, so do -2 = -1-3 NOT -3-1
            y1 = sub1(pc)
            if y1[0] == "C" and all (y == "0" for y in y1[1:]):
                # since C lacks the -3 direction, 
                # but we are subtracting 2 from something on the edge,
                # we can use p-3+3 due to refraction
                res = add3(sub3(pc))
                # raise ValueError(f"ran across pc-1 = C0* in input {pc}{x:+}, will get pc-1-3 = None, need to fix")
            else:
                y13 = sub3(y1)
                res = y13
            # print(f"case: on edge CA, x=-2; {res=}")
        elif on_edge_DB and x == -1:
            if all(y == "0" for y in tail):
                # we will get extraneous solution where p-1 = p-2, but p-3 should be undefined
                res = None
                # print(f"case: on edge DB, x=-1, tail all zero; {res=}")
            else:
                new_head = "F"
                new_tail = replace_zeros_with_twos(tail)
                res = new_head + new_tail
                # print(f"case: on edge DB, x=-1, tail not all zero; {res=}")
        elif on_edge_DB and x == -2:
            # do the refracting last, so do -2 = -3-1 NOT -1-3
            y3 = sub3(pc)
            if y3[0] == "D" and all (y == "0" for y in y3[1:]):
                # since D lacks the -1 direction, 
                # but we are subtracting 2 from something on the edge,
                # we can use p-1+1 due to refraction
                res = add1(sub1(pc))
                # raise ValueError(f"ran across pc-3 = D0* in input {pc}{x:+}, will get pc-3-1 = None, need to fix")
            y31 = sub1(y3)
            res = y31
            # print(f"case: on edge DB, x=-2; {res=}")

        elif x == -2:
            # general case for -2, no refraction
            y1 = sub1(pc)
            y13 = sub3(y1)
            y3 = sub3(pc)
            y31 = sub1(y3)
            # print(f"{pc} -1-3 = {y13} ; {pc} -3-1 = {y31}")
            assert y13 == y31
            res = y13
            # print(f"case: x=-2, not on reversed edge; {res=}")
        else:
            direction_digit = abs(x)
            plus = x > 0
            overflow, new_tail = increment_binary_code(tail, direction_digit, plus)
            
            # overflow is used to change watershed by applying to the point letter
            if overflow == 0:
                new_head = head
            elif overflow == 1:
                assert plus, "shouldn't get positive overflow from subtracting"
                new_head = add_direction_to_point_code(head, x, fix_edge_polarity=False)
            elif overflow == -1:
                assert not plus, "shouldn't get negative overflow from adding"
                new_head = add_direction_to_point_code(head, x, fix_edge_polarity=False)
            else:
                raise ValueError(overflow)
            
            res = new_head + new_tail
            if res[0] in ["A", "B"]:
                # we moved off the left or bottom edge of CD peel
                reference_peel = "KL"
            # print(f"case: {x=}, not on reversed edge; {res=}")

    # only fix the edge polarity at the very final result, 
    # but before reapplying offset (so we still know reference peel is whatever we set it to, usually CD)
    if res is not None and fix_edge_polarity and point_code_is_in_reversed_polarity_encoding(res):
        res = correct_reversed_edge_polarity(res, reference_peel)

    if res not in [None, "?"]:
        pc = apply_peel_offset(pc, peel_offset)
        res = apply_peel_offset(res, peel_offset)
        assert len(res) == len(pc), f"need same #iterations in result but got {pc} {x:+} = {res}"
    # print(f"result: {pc} {x:+} = {res}\n")
    return res

    # head = pc[0]
    # tail = pc[1:]
    # init = pc[:-1]
    # last = pc[-1]
    # pas = NORTHERN_RING
    # pbs = SOUTHERN_RING
    # pa = head in pas
    # pb = head in pbs
    # ps = pas if pa else pbs if pb else None
    # ps_other = pbs if pa else pas if pb else None
    # pi = ps.index(head)
    # p_plus = ps[(pi + 1) % 5]  # for peel shift
    # p_minus = ps[(pi - 1) % 5]  # for peel shift
    # p_other = ps_other[pi]  # switch to top ring or bottom ring (opposite of current one)
    # p_plus_other = ps_other[(pi + 1) % 5]
    # p_minus_other = ps_other[(pi - 1) % 5]

    # if head in ["A", "B"]:
    #     raise ValueError("can't travel direction from pole")
    # return res


add1 = lambda pc: add_direction_to_point_code(pc, 1, fix_edge_polarity=False)
add3 = lambda pc: add_direction_to_point_code(pc, 3, fix_edge_polarity=False)
sub1 = lambda pc: add_direction_to_point_code(pc, -1, fix_edge_polarity=False)
sub3 = lambda pc: add_direction_to_point_code(pc, -3, fix_edge_polarity=False)


def verify_valid_direction(pc, x):
    # check that we are moving in a valid direction
    # since things like A2 are ill-defined (even when edge is reverse-encoded)
    head = pc[0]
    tail = pc[1:]
    if head == "A":
        assert all(y in ["0", "3"] for y in tail), f"invalid point code {pc}"
        if all(y == "0" for y in tail):
            allowed_directions = [3, -1]  # can only go right or down from top left corner
        else:
            allowed_directions = [3, -1, -2, -3]  # cannot go left from left edge
    elif head == "B":
        assert all(y in ["0", "1"] for y in tail), f"invalid point code {pc}"
        if all(y == "0" for y in tail):
            allowed_directions = [1, -3]  # can only go left or up from bottom right corner
        else:
            allowed_directions = [1, 2, 3, -3]  # cannot go right from right edge
    else:
        allowed_directions = [1, 2, 3, -1, -2, -3]
    assert x in allowed_directions, f"direction {x} from {pc=} is not allowed"


def increment_binary_code(tail, direction_digit, plus=True):
    # things like adding 1 or 3 to C102231
    # print(f"increment binary: {tail=}, {direction_digit=}, {plus=}")
    if direction_digit == 1:
        colors = [[0, 1], [3, 2]]
    elif direction_digit == 3:
        colors = [[0, 3], [1, 2]]
    else:
        raise ValueError(direction_digit)
    
    lst = [int(x) for x in tail]
    lst_colors = []
    lst_bits = []
    for x in lst:
        # choose which "color" (digit pair) this digit is in
        # and then choose whether it corresponds to a 0 or a 1 in the binary arithmetic
        if x in colors[0]:
            this_color = colors[0]
        elif x in colors[1]:
            this_color = colors[1]
        else:
            raise ValueError(x)
        this_bit = this_color.index(x)
        lst_colors.append(this_color)
        lst_bits.append(this_bit)
    # print(f"{lst_colors=}, {lst_bits=}")

    overflow, new_bits = binary_up_one(lst_bits) if plus else binary_down_one(lst_bits)
    # print(f"{new_bits=}")
    
    # now get the value within the right color for each bit
    new_lst = []
    for this_color, this_new_bit in zip(lst_colors, new_bits):
        new_digit = this_color[this_new_bit]
        new_lst.append(new_digit)
    new_tail = "".join(str(x) for x in new_lst)
    # print(f"{new_tail=}")

    return overflow, new_tail


def binary_up_one(bits):
    # add 1, make overflow 1 if we rolled over to 0, 0 otherwise
    if all(b == 1 for b in bits):
        overflow = 1
        new_bits = [0] * len(bits)
    else:
        still_adding = True
        n = len(bits)
        new_bits = [None for i in range(n)]
        for i in range(n):
            j = -(i+1)
            b = bits[-(i+1)]
            if still_adding:
                if b == 1:
                    new_bits[j] = 0
                    # and we're still adding, carrying the one to the next place
                elif b == 0:
                    new_bits[j] = 1
                    still_adding = False
            else:
                new_bits[j] = b
        assert not still_adding, "should not have overflowed"
        overflow = 0

    return overflow, new_bits


def binary_down_one(bits):
    # subtract 1, make overflow -1 if we rolled down past 0, 0 otherwise
    if all(b == 0 for b in bits):
        overflow = -1
        new_bits = [1] * len(bits)
    else:
        overflow, new_bits_flipped = binary_up_one(flip_bits(bits))
        assert overflow == 0, "should not have overflowed"
        new_bits = flip_bits(new_bits_flipped)
    return overflow, new_bits


def flip_bits(bits):
    return [1 - b for b in bits]


def normalize_peel(point_code):
    head = point_code[0]
    tail = point_code[1:]
    assert head in list("ABCDEFGHIJKL")
    if head in ["A", "B"]:
        assert len(head) == 1, f"got pole-child code, which should not happen: {point_code}"
        cd_code, peel_offset = point_code, 0
    else:
        if head in NORTHERN_RING:
            peel_offset = NORTHERN_RING.index(head)
            cd_code = "C" + tail
        else:
            assert head in SOUTHERN_RING
            peel_offset = SOUTHERN_RING.index(head)
            cd_code = "D" + tail
    # print(f"{point_code=}, {cd_code=}, {peel_offset=}")
    return cd_code, peel_offset


def apply_peel_offset(pc, peel_offset):
    # print(f"applying peel offset {peel_offset} to {pc}")
    if peel_offset == 0 or pc[0] in ["A", "B"]:
        return pc
    head = pc[0]
    tail = pc[1:]

    if head in NORTHERN_RING:
        ps = NORTHERN_RING
    elif head in SOUTHERN_RING:
        ps = SOUTHERN_RING
    else:
        raise ValueError(f"invalid original ancestor: {head}, from cd_code {pc}, peel_offset {peel_offset}")
    
    new_head = ps[(ps.index(head) + peel_offset) % 5]
    return new_head + tail


def replace_zeros_with_twos(tail):
    res = ""
    for c in tail:
        if c == "0":
            res += "2"
        else:
            assert c == "1" or c == "3", f"shouldn't be replacing 0->2 in string {tail}"
            res += c
    return res
