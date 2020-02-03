import networkx as nx
import graphviz
import matplotlib.pyplot as plt


if __name__ == "__main__":
    input_fp = "GrammaticalizationPaths.txt"
    with open(input_fp) as f:
        lines = f.readlines()
    edges = [l.strip().split("\t") for l in lines]
    assert all(len(e) == 2 for e in edges)
    sources = [e[0] for e in edges]
    destinations = [e[1] for e in edges]

    # g = nx.DiGraph()
    # for n in set(sources) | set(destinations):
    #     g.add_node(n)
    # for e in edges:
    #     g.add_edge(e[0], e[1])
    # nx.draw(g, with_labels=True)
    # plt.show()

    dot = graphviz.Digraph()
    nodes = set(sources) | set(destinations)
    print("\n".join(sorted(nodes)))
    for node in nodes:
        dot.node(node, node)  # name, label
    for e in edges:
        dot.edge(e[0], e[1])
    dot.render('GrammaticalizationPaths', format="png", view=True)
