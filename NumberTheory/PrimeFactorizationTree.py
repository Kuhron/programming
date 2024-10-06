import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sympy import nextprime, prime


def get_tree_graph(n_primes, n_mults, n_levels):
    # n_primes is the primes on the stem, from which multiples branch
    g = nx.Graph()
    primes = [prime(n) for n in range(1, n_primes+1)]  # sympy.prime() is 1-indexed
    g.add_node(primes[0])
    g.nodes[primes[0]]["is_prime"] = True
    g.nodes[primes[0]]["can_mult"] = True
    for n in range(1, n_primes):
        p = primes[n-1]
        q = primes[n]
        g.add_edge(p,q)
        assert g.nodes[p]["is_prime"] is True
        assert g.nodes[p]["can_mult"] is True
        g.nodes[q]["is_prime"] = True
        g.nodes[q]["can_mult"] = True
    for level_i in range(n_levels):
        g = add_level(g, n_mults)
    return g


def add_level(g, n_mults):
    mults = [prime(n) for n in range(1, n_mults+1)]
    nodes_to_mult_from = [node_label for node_label in g.nodes if g.nodes[node_label]["can_mult"] is True]
    for node_label in nodes_to_mult_from:
        node = g.nodes[node_label]
        for m in mults:
            val = node_label * m
            g.add_edge(node_label, val)
            g.nodes[val]["is_prime"] = False
            g.nodes[val]["can_mult"] = True
        node["can_mult"] = False  # now we've added everything we ever will
    return g


def get_factor_string(a,b):
    # return the ratio between a and b as an integer >= 1
    assert type(a) is int and type(b) is int
    if a <= b:
        if b % a != 0:
            return ""
        r = b//a
    else:
        if a % b != 0:
            return ""
        r = a//b
    return str(r)


def draw_tree(n_primes, n_mults, n_levels):
    g = get_tree_graph(n_primes, n_mults, n_levels)
    plt.figure()
    pos = nx.spring_layout(g)
    nx.draw(g, pos, node_color="white", with_labels=True)
    edge_labels = {(a,b): get_factor_string(a,b) for a,b in g.edges}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="red")
    plt.show()


if __name__ == "__main__":
    n_primes = 17
    n_mults = 5
    n_levels = 3

    draw_tree(n_primes, n_mults, n_levels)

