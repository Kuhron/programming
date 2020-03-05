from sympy import divisor_count
import networkx as nx
import matplotlib.pyplot as plt


def make_tree(n):
    g = nx.DiGraph()
    for x in range(2, n+1):
        g.add_node(x)
        y = divisor_count(x)
        g.add_edge(x, y)
    nx.draw(g, with_labels=True)
    plt.show()


if __name__ == "__main__":
    make_tree(1000)
