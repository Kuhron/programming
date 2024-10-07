import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sympy import nextprime, prime, factorint
import math


def get_tree_graph(paths_dict):
    base_primes = sorted(paths_dict.keys())
    g = get_graph_base_primes_only(base_primes)
    for p in base_primes:
        g = add_paths_from_node(g, starting_node_label=p, paths_dict_from_node=paths_dict[p])

    return g


def get_graph_base_primes_only(base_primes):
    g = nx.Graph()
    g.add_node(base_primes[0])
    for p_i in range(1, len(base_primes)):
        p = base_primes[p_i-1]
        q = base_primes[p_i]
        g.add_edge(p,q)
        g.edges[(p,q)]["factor"] = None
    return g


def add_paths_from_node(g, starting_node_label, paths_dict_from_node):
    if len(paths_dict_from_node) == 0:
        return g
    branch_primes = sorted(paths_dict_from_node.keys())
    node = g.nodes[starting_node_label]

    n = starting_node_label
    for m in branch_primes:
        new_val = n*m
        new_val_already_exists = new_val in g.nodes
        g.add_edge(n, new_val)
        g.edges[(n, new_val)]["factor"] = m

        if not new_val_already_exists:
            g = add_paths_from_node(g, starting_node_label=new_val, paths_dict_from_node=paths_dict_from_node[m])

    for m_i in range(1, len(branch_primes)):
        p = branch_primes[m_i-1]
        q = branch_primes[m_i]
        g.add_edge(n*p, n*q)
        g.edges[(n*p, n*q)]["factor"] = None

    return g


def get_factorization_paths_all(base_primes, branch_primes, n_levels):
    if n_levels == 0:
        return {p: {} for p in base_primes}
    else:
        sub_dict = get_factorization_paths_all(base_primes=branch_primes, branch_primes=branch_primes, n_levels=n_levels-1)
        return {p: sub_dict for p in base_primes}


def get_factorization_paths_sorted(base_primes, branch_primes, n_levels):
    if n_levels == 0:
        return {p: {} for p in base_primes}
    else:
        d = {}
        for p in base_primes:
            new_branch_primes = [x for x in branch_primes if x >= p]
            sub_dict = get_factorization_paths_sorted(base_primes=new_branch_primes, branch_primes=new_branch_primes, n_levels=n_levels-1)
            d[p] = sub_dict
        return d


def get_factorization_paths_sorted_in_range(n_min, n_max):
    paths_dict = {}
    for n in range(n_min, n_max+1):
        d = factorint(n)
        path = factor_dict_to_sorted_list(d)
        paths_dict = add_path_list_to_paths_dict(path, paths_dict)
    return paths_dict


def add_path_list_to_paths_dict(lst, d):
    current_dict = d  # can mutate the sub-dicts by reference
    while len(lst) > 0:
        p = lst[0]
        if p not in current_dict:
            current_dict[p] = {}
        current_dict = current_dict[p]
        lst = lst[1:]
    return d


def factor_dict_to_sorted_list(d):
    lst = []
    for p, n in sorted(d.items()):
        lst += [p] * n
    return lst


def if_none(x, default):
    if x is None:
        return default
    return x


def draw_tree(paths_dict):
    g = get_tree_graph(paths_dict)
    plt.figure()
    pos = nx.spring_layout(g)
    edge_colors = ["#00cccc" if e["factor"] is None else "black" for (a,b), e in g.edges.items()]
    nx.draw(g, pos, node_color="white", edge_color=edge_colors, with_labels=True)
    edge_labels = {(a,b): if_none(e["factor"], "") for (a,b), e in g.edges.items()}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="red")
    plt.show()


if __name__ == "__main__":
    # n_base_primes = 11
    # n_branch_primes = 11
    # n_levels = 3

    # base_primes = [prime(n) for n in range(1, n_base_primes+1)]  # sympy.prime() is 1-indexed
    # branch_primes = [prime(n) for n in range(1, n_branch_primes+1)]

    # paths_dict = get_factorization_paths_all(base_primes, branch_primes, n_levels)
    # paths_dict = get_factorization_paths_sorted(base_primes, branch_primes, n_levels)
    paths_dict = get_factorization_paths_sorted_in_range(n_min=2, n_max=999)

    draw_tree(paths_dict)

    # questions about this connection between graph theory and number theory
    # - what are the properties of the infinite graph connecting integers by prime factors?
    # - how are the properties changed (e.g. number of components, degree statistics) when we restrict which connections we allow in the graph? e.g. only factorizations sorted in some order (e.g. increasing only, decreasing only, alternating between smallest and biggest remaining factor like [2, 13, 2, 11, 3, 7]), only certain subset of the primes as factors, only certain subset of the integers being factored?


