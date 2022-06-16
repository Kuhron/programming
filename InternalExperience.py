# try to make a mind that will receive input from the environment
# and it will have an internal state like a brain does
# brains don't ever reach stasis like a program would
# they always find something to think about, something new to do
# they go through phases, partially because of environmental changes
# but partially because of chaos within
# want to emulate this


import random
import time
import networkx as nx
import matplotlib.pyplot as plt

# try 3 input streams, no output streams (it just senses stuff and thinks about it, can't affect its environment)
# want internal self-editing to be possible, it can rewire itself
# want mechanisms to make stasis cause it to change something so stasis doesn't continue
# ideally the brain is asynchronous, so the environment can just keep throwing stuff at it regardless of whether it hangs on processing
# input streams from environment can be random, but brain should be deterministic given the input (so I can't create internal change using random noise from within)


class Brain:
    def __init__(self):
        self.graph = nx.Graph()
        for i in range(3):  # number of inputs
            self.graph.add_node(i+1)
            self.graph.nodes[i+1]["weight"] = 0
        self.reserved_nodes = [1, 2, 3]  # brain is not allowed to remove these (the sensory inputs)

    def draw(self):
        g = self.graph
        # only label certain nodes (the sensors): https://stackoverflow.com/questions/14665767/networkx-specific-nodes-labeling
        labels = {}
        colors = []
        for node in g.nodes():
            if node in self.reserved_nodes:
                # set the node name as the key and the label as its value
                labels[node] = node
                colors.append("red")
            else:
                colors.append("blue")
        # set the argument 'with labels' to False so you have unlabeled graph
        pos = nx.spring_layout(g)
        nx.draw(g, pos, node_color=colors, with_labels=False)
        # now only add labels to the nodes you require
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()

    def receive_input(self, indices, values):
        # input sensor #{index} received value
        # should do this even for None inputs, so brain knows time passes

        g = self.graph
        next_node_num = len(g.nodes)
        for index, val in zip(indices, values):
            neighbors = g.neighbors(index)
            # choose a neighbor to do something to
            chosen_neighbor = None
            for node in neighbors:
                w = g.nodes[node]["weight"]
                if val is None or w < val:
                    chosen_neighbor = node
                    break
            if chosen_neighbor is None:
                # make a new neighbor to use
                g.add_node(next_node_num)
                g.nodes[next_node_num]["weight"] = 0
                g.add_edge(index, next_node_num)
                chosen_neighbor = next_node_num
                next_node_num += 1
            # now we have a neighbor connected to this input sensor, change its weight
            w = g.nodes[chosen_neighbor]["weight"]
            if val is None:
                val = sum(g.nodes[node]["weight"] for node in g.neighbors(chosen_neighbor))
            g.nodes[chosen_neighbor]["weight"] = int(round((w + val) / 2))

        print("nodes", g.nodes)
        print("edges", g.edges)
        print("weights", [g.nodes[node]["weight"] for node in g.nodes])


def alternating_sum(xs, first_positive=True):
    p = 1 if first_positive else -1
    s = 0
    for x in xs:
        s += x * p
        p *= -1
    return s


def input_generating_machine(p_switch, last_n_window, max_error):
    is_on = random.choice([True, False])
    last_n = []
    while True:
        if random.random() < p_switch:
            is_on = not is_on
        if is_on:
            s = alternating_sum(last_n)
            error = random.randint(-max_error, max_error)
            val = s + error
            yield val
            last_n = last_n[1:] + [val]
        else:
            yield None  # don't give any input
            last_n = last_n[1:]  # don't put a None on it, but do remove the oldest one


if __name__ == "__main__":
    inp1 = input_generating_machine(p_switch=0.005, last_n_window=5, max_error=5)
    inp2 = input_generating_machine(p_switch=0.015, last_n_window=12, max_error=3)
    inp3 = input_generating_machine(p_switch=0.0001, last_n_window=70, max_error=10)

    brain = Brain()

    val_to_str = lambda val, n=6: " "*n if val is None else str(val).rjust(n)
    t_i = 0
    while True:
        val1 = next(inp1)
        val2 = next(inp2)
        val3 = next(inp3)
        print(f"t = {t_i}:", val_to_str(val1), val_to_str(val2), val_to_str(val3))
        brain.receive_input([1, 2, 3], [val1, val2, val3])
        if t_i > 0 and t_i % 100 == 0:
            brain.draw()

        t_i += 1
        time.sleep(0.001)

