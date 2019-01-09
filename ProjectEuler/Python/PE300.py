def get_folding_coordinate_lists(n_points, starting_point=(0, 0), used_points=set()):
    assert n_points > 0
    if n_points == 1:
        return [[starting_point]]
    else:
        result = []
        # can go four cardinal directions, then recursively add paths from length n-1 starting at that next point
        directions = [(0, -1)] if used_points == set() else [(1, 0), (0, 1), (-1, 0), (0, -1)]  # if it's the first step, go down (avoid doing 4x as much work, don't worry about reflections right now)
        for direction in directions:
            new_point = (starting_point[0] + direction[0], starting_point[1] + direction[1])
            if new_point in used_points:
                continue  # don't add this result to the list
                # if we are in a recursive call then anything leading to this point 
                # (forcing the sequence to terminate early) should not be included
            else:  # new point is the starting point for the recursive call
                result += [[starting_point] + lst for lst in get_folding_coordinate_lists(n_points - 1, new_point, used_points | {starting_point})]
        return result
            

def print_folding_coordinate_list(lst):
    assert len(lst) <= 36
    s = "0123456789abcdefghijklmnopqrstuvwxyz"
    min_x = min(p[0] for p in lst)
    max_x = max(p[0] for p in lst)
    min_y = min(p[1] for p in lst)
    max_y = max(p[1] for p in lst)
    print_str = ""
    for y in range(max_y, min_y - 1, -1):  # print y backwards so down is on the bottom
        for x in range(min_x, max_x + 1):
            if (x, y) in lst:
                print_str += s[lst.index((x, y))]
            else:
                print_str += "-"
        print_str += "\n"
    print(print_str)


def get_all_protein_strings(length):
    assert length > 0
    if length == 1:
        return ["H", "P"]
    else:
        prev = get_all_protein_strings(length - 1)
        return ["H" + x for x in prev] + ["P" + x for x in prev]

def count_hh_contacts(protein_str, coordinate_lst):
    def get_d_and_h_pts():
        d = {}
        h_pts = []
        for point, amino in zip(coordinate_lst, protein_str):
            d[point] = amino
            if amino == "H":
                h_pts.append(point)
        return d, h_pts
    d, h_pts = get_d_and_h_pts()  # put it in a function for profiling

    def get_result():
        result = 0
        # contacts = set()
        for p0 in h_pts:
            for direction in [(1, 0), (0, -1)]:  # only go right and down so we don't double count
                p1 = (p0[0] + direction[0], p0[1] + direction[1])
                if p1 in d and d[p1] == "H":
                    # contacts.add((p0[0], p0[1], p1[0], p1[1]))
                    result += 1
        return result

        # for p1 in h_pts:
        #     if p0 < p1:
        #         dx = p0[0] - p1[0]
        #         dy = p0[1] - p1[1]
        #         border = (dx == 0 and dy == 1) or (dx == 1 and dy == 0) or (dx == 0 and dy == -1) or (dx == -1 and dy == 0)
        #         if border:
        #             contacts.add((p0[0], p0[1], p1[0], p1[1]))
    # return len(contacts)
    result = get_result()
    return result


def get_max_hh_contacts(protein_str, coordinate_lists):
    result = 0
    for coordinate_lst in coordinate_lists:
        result = max(result, count_hh_contacts(protein_str, coordinate_lst))
    return result


if __name__ == "__main__":
    # test
    # for lst in get_folding_coordinate_lists(7):
    #     print(lst)
    #     print_folding_coordinate_list(lst)

    length = 15
    protein_strs = get_all_protein_strings(length)
    coordinate_lists = get_folding_coordinate_lists(length)
    n_strs = len(protein_strs)
    assert n_strs == 2**length
    hh_contact_counts = []
    for i, protein_str in enumerate(protein_strs):
        print("{:.1f}% done ({} of {})".format(100*i/n_strs, i, n_strs),end="\r")
        hh_contact_counts.append(get_max_hh_contacts(protein_str, coordinate_lists))
    print()
    print(sum(hh_contact_counts), n_strs, sum(hh_contact_counts)/n_strs)
