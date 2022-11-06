raise Exception("deprecated")


# def parse_adjacency_line(l):
#     pi, neighbors_str = l.strip().split(":")
#     pi = int(pi)
#     neighbors = [int(x) for x in neighbors_str.split(",")]
#     return pi, neighbors


# def parse_position_line(l):
#     pi, rest = l.strip().split(":")
#     pi = int(pi)
#     xyz_str, latlon_str = rest.split(";")
#     xyz = [float(x) for x in xyz_str.split(",")]
#     assert len(xyz) == 3
#     latlon = [float(x) for x in latlon_str.split(",")]
#     assert len(latlon) == 2
#     return pi, xyz, latlon


# def get_adjacency_line(point_number, ordered_neighbor_point_numbers):
#     s1 = str(point_number)
#     s2 = ",".join(str(x) for x in ordered_neighbor_point_numbers)
#     return s1 + ":" + s2 + "\n"


# def get_position_line(point_number, xyz, latlon):
#     s1 = str(point_number)
#     s2 = ",".join(str(x) for x in xyz)
#     s3 = ",".join(str(x) for x in latlon)
#     return s1 + ";" + s2 + ";" + s3 + "\n"


# def parse_adjacency_memo_file(memo_fp):
#     # format: each line is index:neighbor_list (comma separated point numbers)
#     # e.g. 0:1752,1914,2076,2238,2400
#     with open(memo_fp) as f:
#         notify_memo_accessed(memo_fp)
#         lines = f.readlines()
#     d = {}
#     for l in lines:
#         pi, neighbors = parse_adjacency_line(l)
#         d[pi] = neighbors
#     return d


# def parse_position_memo_file(memo_fp):
#     # format: each line is index:xyz;latlon (xyz and latlon are comma-separated)
#     # e.g. 0:6.123233995736766e-17,0.0,1.0;90,0
#     with open(memo_fp) as f:
#         notify_memo_accessed(memo_fp)
#         lines = f.readlines()
#     d = {}
#     for l in lines:
#         pi, xyz, latlon = parse_position_line(l)
#         d[pi] = {"xyz":xyz, "latlondeg":latlon}
#     return d


# def write_initial_memo_files():
#     # ordered_points, adjacencies_by_point_index = get_starting_points()
#     ordered_points = STARTING_POINTS_ORDERED
#     adjacencies_by_point_index = STARTING_POINTS_ADJACENCY

#     s_adj = ""
#     s_pos = ""
#     for pi, p in enumerate(ordered_points):
#         adj = adjacencies_by_point_index[pi]
#         l_adj = get_adjacency_line(pi, adj)
#         l_pos = get_position_line(pi, p.xyz(), p.latlondeg())
#         assert l_adj[-1] == l_pos[-1] == "\n"
#         s_adj += l_adj
#         s_pos += l_pos
#     with open(get_adjacency_memo_fp(n_iterations=0), "w") as f:
#         f.write(s_adj)
#     with open(get_position_memo_fp(n_iterations=0), "w") as f:
#         f.write(s_pos)
#     print("initial memo files written")


# def notify_memo_accessed(memo_fp):
#     # depending what I want at the time, maybe do nothing, maybe just print that it was accessed, or maybe raise exception if I'm trying to avoid any memoization at all
#     # pass
#     print("memo accessed: {}".format(memo_fp))
#     # raise RuntimeError("memo accessed but shouldn't be: {}".format(memo_fp))  # advantage here that it will show the call stack


# def get_adjacency_memo_fp(n_iterations):
#     return "/home/wesley/programming/Mapping/MemoIcosa/MemoIcosaAdjacency_Iteration{}.txt".format(n_iterations)


# def get_position_memo_fp(n_iterations):
#     return "/home/wesley/programming/Mapping/MemoIcosa/MemoIcosaPosition_Iteration{}.txt".format(n_iterations)


# def get_adjacency_memo_dict(n_iterations):
#     return parse_adjacency_memo_file(get_adjacency_memo_fp(n_iterations))


# def get_position_memo_dict(n_iterations):
#     return parse_position_memo_file(get_position_memo_fp(n_iterations))


# def get_specific_adjacencies_from_memo(point_numbers, n_iterations):
#     verify_valid_point_numbers(point_numbers, n_iterations)
#     memo_fp = get_adjacency_memo_fp(n_iterations)
#     # print("getting adjacency memo for iteration {}: {}".format(n_iterations, memo_fp))
#     with open(memo_fp) as f:
#         notify_memo_accessed(memo_fp)
#         lines = f.readlines()
#     lines = [lines[i] for i in point_numbers]
#     d = {}
#     for point_number, l in zip(point_numbers,lines):
#         pi, neighbors = parse_adjacency_line(l)
#         assert pi == point_number
#         d[pi] = neighbors
#     # print("returning {}".format(d))
#     return d


# def get_specific_adjacency_from_memo(point_number, n_iterations):
#     return get_specific_adjacencies_from_memo([point_number], n_iterations)[point_number]


# def get_specific_positions_from_memo(point_numbers, n_iterations):
#     verify_valid_point_numbers(point_numbers, n_iterations)
#     memo_fp = get_position_memo_fp(n_iterations)
#     with open(memo_fp) as f:
#         notify_memo_accessed(memo_fp)
#         lines = f.readlines()
#     lines = [lines[i] for i in point_numbers]
#     d = {}
#     for point_number, l in zip(point_numbers,lines):
#         pi, xyz, latlon = parse_position_line(l)
#         assert pi == point_number
#         d[pi] = {"xyz":xyz, "latlondeg":latlon}
#     return d


# def get_specific_position_from_memo(point_number):
#     n_iterations = get_iterations_needed_for_point_number(point_number)
#     return get_specific_positions_from_memo([point_number], n_iterations)[point_number]


