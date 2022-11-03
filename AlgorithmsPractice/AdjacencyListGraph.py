import networkx as nx


class AdjacencyListGraph:
    # store nodes by index primarily
    # only care about node labels on the outside (how user interfaces with this class), not in internal implementation

    def __init__(self, nodes=None):
        self.adjacency_list = []
        self.node_labels = []
        self.node_index_by_label = {}
        self.n_nodes = 0

        if nodes is not None:
            for node in nodes:
                self.add_node(node)

    def add_node(self, label):
        self.adjacency_list.append([])
        self.node_labels.append(label)
        self.node_index_by_label[label] = self.n_nodes  # since number from 0
        self.n_nodes += 1

    def add_edge_by_labels(self, label1, label2):
        i1 = self.node_index_by_label[label1]
        i2 = self.node_index_by_label[label2]
        self.add_edge_by_indices(i1, i2)

    def add_edge_by_indices(self, i1, i2):
        if self.has_edge_by_indices(i1, i2):
            return

        self.adjacency_list[i1].append(i2)
        self.adjacency_list[i2].append(i1)

    def has_edge_by_indices(self, i1, i2):
        adj = self.adjacency_list[i1]
        for i in adj:
            if i == i2:
                return True
        return False

    def to_networkx(self):
        # for making plotting easier
        g = nx.Graph()
        for node in self.node_labels:
            g.add_node(node)
        for i in range(self.n_nodes):
            ni = self.node_labels[i]
            for j in self.adjacency_list[i]:
                nj = self.node_labels[j]
                g.add_edge(ni, nj)
        return g
