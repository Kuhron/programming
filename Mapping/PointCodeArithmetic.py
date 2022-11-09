# implementing the "arithmetic" on point code strings
# to try to remove the need for recursion as much as possible in computing adjacency


# equations about how this arithmetic works
# where r, s = some string made of {0,1,2,3};
#       P, Q = initial icosa points;
#       PA, PB = initial icosa points on the upper or lower ring, respectively
#       x = direction in {1,2,3};
#       (+), (-) = peel shift ahead (C->E) or back (C->K), respectively.
#           (note that peel shift directions are backwards from addition/subtraction,
#           i.e. adding 1/2/3 yields no peel shift or a backward peel shift
#           and subtracting 1/2/3 yields no peel shift or a forward peel shift)
# there might be a way to unify some of this some more, but I think it works for now

# identity 1: Px - x = P0  (similarly Prx - x = Pr0)
# identity 2: P1 + 3 = P3 + 1 = P2 ; => P2 - 3 = P1 ; P2 - 1 = P3
# identity 3: PA1 - 3 = None ; PB3 - 1 = None
# identity 4: PA1 + 1 = A0 ; PB3 + 3 = B0
# identity 5: PA1 + 2 = (-)PA1 ; PA1 - 2 = (+)PA1 ; PB3 + 2 = (-)PB3 ; PB3 - 2 = (+)PB3
# identity 6: PB1 + 1 = PA0
# identity 7: P2 + 2 = (-)P0
# identity 8: PA2 + 1 = (-)PA1 ; PB2 + 3 = (-)PB3
# identity 9: PA3 + 3 = (-)PB0
# identity 10: PA2 + 3 = PA3 + 2 = (-)PB1 ; => PB1 - 2 = (+)PA3 ; PB1 - 3 = (+)PA2
# identity 11: PB1 + 2 = PB2 + 1 = PA3 ; => PA3 - 1 = PB2 ; PA3 - 2 = PB1
# identity 12: lim {PA1, PA11, PA111, ...} = A ; lim {PB3, PB33, PB333, ...} = B
# identity 13: if Pr + x = Ps, then Prx + x = Ps0
# identity 14: if Pr + x = Ps, then Pxr + x = P0s

# use Haskell convention of lst = [head] + [tail ...] = [init ...] + [last]


def add_direction_to_point_code(pc, x):
    head = pc[0]
    tail = pc[1:]
    init = pc[:-1]
    last = pc[-1]
    pas = ["C", "E", "G", "I", "K"]
    pbs = ["D", "F", "H", "J", "L"]
    pa = head in pas
    pb = head in pbs
    ps = pas if pa else pbs if pb else None
    ps_other = pbs if pa else pas if pb else None
    pi = ps.index(head)
    p_plus = ps[(pi + 1) % 5]  # for peel shift
    p_minus = ps[(pi - 1) % 5]  # for peel shift
    p_other = ps_other[pi]  # switch to top ring or bottom ring (opposite of current one)
    p_plus_other = ps_other[(pi + 1) % 5]
    p_minus_other = ps_other[(pi - 1) % 5]

    if head in ["A", "B"]:
        raise ValueError("can't travel direction from pole")

    if (last == "1" and x == "3") or (last == "3" and x == "1"):
        # identity 2
        res = init + "2"
    elif pa and all(y == "1" for y in tail) and x == "1":
        # identity 4
        res = "A" + "0" * len(tail)
    elif pb and all(y == "3" for y in tail) and x == "3":
        # identity 4
        res = "B" + "0" * len(tail)
    elif (pa and last == "1" and x == "2") or (pb and last == "3" and x == "2"):
        # identity 5
        res = p_minus + tail
    elif pb and all(y == "1" for y in tail) and x == "1":
        # identity 6
        res = p_other + "0" * len(tail)
    elif pa and all(y == "3" for y in tail) and x == "3":
        # identity 9: PA3 + 3 = (-)PB0
        res = p_minus_other + "0" * len(tail)
    else:
        # raise NotImplementedError
        return "?"
    
    assert len(res) == len(pc), f"need same #iterations in result but got {pc} + {x} = {res}"
    return res


def subtract_direction_from_point_code(pc, x):
    head = pc[0]
    tail = pc[1:]
    init = pc[:-1]
    last = pc[-1]
    pas = ["C", "E", "G", "I", "K"]
    pbs = ["D", "F", "H", "J", "L"]
    pa = head in pas
    pb = head in pbs
    ps = pas if pa else pbs if pb else None
    p_plus = ps[(1 + ps.index(head)) % 5]  # for peel shift
    p_minus = ps[(-1 + ps.index(head)) % 5]  # for peel shift

    if head in ["A", "B"]:
        raise ValueError("can't travel direction from pole")

    if pc[-1] == x:
        # identity 1
        # pc = P alpha x
        prefix = pc[:-1]
        res = prefix + "0"
    elif last == "2" and x == "3":
        # identity 2
        res = init + "1"
    elif last == "2" and x == "1":
        # identity 2
        res = init + "3"
    elif (pa and last == "1" and x == "3") or (pb and last == "3" and x == "1"):
        # identity 3
        res = None
    elif (pa and last == "1" and x == "2") or (pb and last == "3" and x == "2"):
        # identity 5
        res = p_plus + tail
    else:
        # raise NotImplementedError
        return "?"

    if res is not None:
        assert len(res) == len(pc), f"need same #iterations in result but got {pc} + {x} = {res}"
    return res

