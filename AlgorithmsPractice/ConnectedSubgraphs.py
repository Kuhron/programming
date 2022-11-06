# problem from my semantic map QP
# given a set of nodes and a set of subsets of those nodes
# make a graph such that each of the subsets is a connected subgraph
# and minimize the number of edges in the whole graph
# this might be NP-hard, not sure


import random
import time
import networkx as nx  # ONLY USE FOR DRAWING, need to implement graph stuff yourself here
import numpy as np
import matplotlib.pyplot as plt

from AdjacencyListGraph import AdjacencyListGraph


def get_node_subsets(nodes, n_subsets):
    # choose some random subsets of the nodes
    n_nodes = len(nodes)
    if n_nodes < 3:
        raise ValueError("too few nodes")
    # each subset has at least 2 elements and at most n_nodes - 1 elements
    # first element choices, second element choices, omitted element choices, powerset of rest
    # n_subsets_possible = (n_nodes) * (n_nodes - 1) * (n_nodes - 2) * (2**(n_nodes-3))
    # if n_subsets > n_subsets_possible ** 0.5:
    #     raise ValueError("too many subsets requested")

    res = set()
    while len(res) < n_subsets:
        mu = max(2, n_nodes ** 0.2)
        prob = mu / n_nodes
        subset = {node for node in nodes if random.random() < prob}
        while len(subset) < 2:
            subset.add(random.choice(nodes))
        while len(subset) >= n_nodes:
            subset.remove(random.choice(subset))
        res.add(tuple(sorted(subset)))
    return sorted(res)


def make_complete_graph(nodes):
    g = AdjacencyListGraph.complete(nodes)
    return g


def make_connected_graph_complete(nodes, subsets):
    # TODO analyze runtime
    if subsets is None:
        return make_complete_graph(nodes)
    g = AdjacencyListGraph(nodes)
    for subset_i, subset in enumerate(subsets):
        # print(f"subset {subset_i}/{len(subsets)}")
        n = len(subset)
        for i in range(n):
            for j in range(i+1, n):
                ni = subset[i]
                nj = subset[j]
                g.add_edge_by_labels(ni, nj)
    # connectedness of subgraphs is guaranteed, don't need to check
    return g


def make_connected_graph_random_edges_no_priority(nodes, subsets):
    # TODO analyze runtime
    g = AdjacencyListGraph()
    raise


def make_connected_graph_random_edges_with_priority(nodes, subsets):
    # subgraphs can vote for the edges they want to add
    g = AdjacencyListGraph()
    raise


def make_connected_graph_random_edge_removal_no_priority(nodes, subsets):
    # TODO analyze runtime
    # start with complete graph and remove edges as long as they don't disconnect a subgraph
    g = make_connected_graph_complete(nodes, subsets)
    raise


def subgraph_is_connected(g, nodes):
    # try doing it without making a separate graph object for the subgraph
    component = get_first_connected_component_indices(g, starting_index=nodes[0], node_indices_to_check=nodes)
    return len(component) == len(nodes)


def is_connected(g):
    # do it myself, don't just call nx.is_connected
    # because I want to play with different ways to check this
    return len(get_first_connected_component_indices(g)) == g.n_nodes


def get_all_connected_component_indices(g):
    node_indices_to_check = list(range(g.n_nodes))
    # keep getting first component until none is left
    components = []
    while len(node_indices_to_check) > 0:
        component = get_first_connected_component_indices(g, node_indices_to_check=node_indices_to_check)
        components.append(component)
        node_indices_to_check = [x for x in node_indices_to_check if x not in component]
    return components


def get_first_connected_component_indices(g, starting_index=None, node_indices_to_check=None):
    node_indices = list(range(g.n_nodes))
    # start at first node in the list
    # go over its neighbors, add them to the group
    # then for each group member you haven't checked yet,
    # do the same (ignoring neighbors that you already know are in the group)
    # once there are no unchecked members, you have all the nodes reachable from the first node
    # the group keeps track of node INDICES not values

    if node_indices_to_check is None:
        node_indices_to_check = list(range(g.n_nodes))
    if starting_index is None:
        starting_index = node_indices_to_check[0]

    indices_in_group = [starting_index]
    nodes_checked = 0
    while nodes_checked < len(indices_in_group):
        # check the first node in the group that we haven't checked yet
        # "check" means look at its neighbors in order and add them to the group
        # don't need to check neighbors with lower indices than the node we're currently on, since this node was added by checking one of those
        node_i = indices_in_group[nodes_checked]  # get the first one we haven't yet checked
        potential_neighbor_indices = node_indices_to_check
        for potential_neighbor_i in potential_neighbor_indices:
            potential_neighbor = node_indices[potential_neighbor_i]
            if g.has_edge_by_indices(node_i, potential_neighbor_i):
                if potential_neighbor_i not in indices_in_group:
                    indices_in_group.append(potential_neighbor_i)
        nodes_checked += 1
    return indices_in_group


def plot_algorithm_performance(function, input_sizes, max_time=None):
    xs = input_sizes
    ys = []
    start_time = time.time()
    for x in xs:
        print(f"timing for {x=}")
        t0 = time.time()
        function(x)
        t1 = time.time()
        dt = t1 - t0
        ys.append(dt)
        print(f"input size {x=} took {dt=} seconds")
        if max_time is not None and t1 - start_time > max_time:
            break
    plt.plot(xs[:len(ys)], ys)
    plt.show()


if __name__ == "__main__":
    def f(n_nodes):
        n_subsets = n_nodes/2
        nodes = list(range(n_nodes))
        g = make_complete_graph(nodes)
        # subsets = get_node_subsets(nodes, n_subsets)
        # g = make_connected_graph_complete(nodes, subsets)
        # components = get_all_connected_component_indices(g)

    # nx.draw(g.to_networkx(), with_labels=True)
    # plt.show()

    plot_algorithm_performance(f, range(100, 10000, 100), max_time=60)

