import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


n = 20
g = nx.Graph()

center_node_int = 0
g.add_node(center_node_int)


for i in range(n):
    node_int = random.choice(g.nodes())

    def is_acceptable(x):
        return (
            x != node_int and
            x not in g.neighbors(node_int) and
            random.random() < 0.5
        )

    options = [x for x in filter(is_acceptable, range(len(g.nodes())))] + [len(g.nodes()) + 1]
    other_node_int = random.choice(options)  # choose a new node with equal probability as any existing node

    edge_tuple = (node_int, other_node_int)
    g.add_edge(*edge_tuple)

assert len(g.edges()) == n

distances = nx.single_source_shortest_path_length(g, center_node_int)

for node_int in g.nodes():
    node = g.node[node_int]
    node["distance"] = distances[node_int]
    print(node_int, node["distance"])

for edge_tuple in g.edges():
    n1, n2 = edge_tuple
    edge = g.edge[n1][n2]
    edge["polar"] = (g.node[n1]["distance"] != g.node[n2]["distance"])
    print(edge_tuple, edge)


nx.draw_networkx(g, with_labels=True)
plt.show()