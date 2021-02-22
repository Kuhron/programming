# generate some random graph in networkx and then just look at the spectral layout to see how cool it is
# according to Daniel Spielman (https://www.youtube.com/watch?v=CDMQR422LGM), this doesn't work well for all graphs, but it's still awesome


import networkx as nx
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_spring_versus_spectral(g):
    pos1 = nx.spring_layout(g)
    pos2 = nx.spectral_layout(g)
    plt.subplot(1,2,1)
    nx.draw(g, pos=pos1)
    plt.subplot(1,2,2)
    nx.draw(g, pos=pos2)
    plt.show()


def get_random_graph(n):
    edge_likelihood = random.uniform(0.1, 0.6)
    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
    for i, j in itertools.combinations(range(n), 2):
        if random.random() < edge_likelihood:
            g.add_edge(i,j)
    # make sure have 1 component
    n_components = nx.number_connected_components(g)
    while n_components > 1:
        i, j = random.choice(list(itertools.combinations(range(n),2)))
        g.add_edge(i,j)
    return g


if __name__ == "__main__":
    n = random.randint(15, 30)
    g = get_random_graph(n)
    # g = nx.path_graph(n)
    plot_spring_versus_spectral(g)

    # discoveries
    # nx.path_graph(5)  # parabola y=x^2

