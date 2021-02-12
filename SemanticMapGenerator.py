import random
import string
import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def get_concept_names():
    with open("cmudict.txt") as f:
        lines = f.readlines()
    names = list(set(l.split()[0] for l in lines))  # some words have more than one line in the cmudict, filter duplicates
    # filter out some crappy ones, e.g. all the ones which are possessive
    names = [x for x in names if not x.endswith("'S") and not x.endswith("'")]
    return names


def create_conceptual_space(n_concepts):
    concepts = random.sample(get_concept_names(), n_concepts)
    g = nx.Graph()
    for c in concepts:
        g.add_node(c)
        g.nodes[c]["concept"] = c
    for a,b in itertools.combinations(concepts, 2):
        g.add_edge(a, b, weight=random.random())
    return g


def get_wordform():
    vowels = "aoeui"
    consonants = [x for x in string.ascii_lowercase if x not in vowels]
    sonorants = "mnlryw"
    c = lambda: random.choice(consonants)
    v = lambda: random.choice(vowels)
    s = lambda: random.choice(sonorants)
    initial = lambda: c() if random.random() < 0.7 else ""
    final = lambda: c() if random.random() < 0.3 else ""
    noninitial_onset = lambda: c()
    nonfinal_coda = lambda: s() if random.random() < 0.2 else ""

    initial_syll = lambda: initial() + v() + nonfinal_coda()
    medial_syll = lambda: noninitial_onset() + v() + nonfinal_coda()
    final_syll = lambda: noninitial_onset() + v() + final()
    sole_syll = lambda: initial() + v() + final()

    w = lambda n: sole_syll() if n <= 1 else initial_syll() + "".join(medial_syll() for n_ in range(n-2)) + final_syll()
    n = random.randint(1, 3)
    return w(n)


def get_n_wordforms(n):
    res = set()
    while len(res) < n:
        res.add(get_wordform())  # filter duplicates because it's a set
    return res


def add_initial_forms(conceptual_space):
    cs = deepcopy(conceptual_space)
    n_concepts = len(cs.nodes)
    wfs = get_n_wordforms(n_concepts)
    for wf, node in zip(wfs, cs.nodes):
        cs.nodes[node]["wordform"] = wf
    return cs


def make_semantic_change(g):
    g = deepcopy(g)
    # select random node, either extend its meaning to a neighbor, or replace it with a novel word
    chosen_node = random.choice(list(g.nodes))  # random.choice(g.nodes) does weird thing where choice gets an int, and tries to access that int index on the iterable, but then g.nodes interprets that int index as a node name and throws KeyError
    extend = random.random() < 0.9
    if extend:
        # extend this node's wordform to one of its neighbors, overwriting whatever was there before
        neighbors = list(g.neighbors(chosen_node))
        # get edge weights to each neighbor
        weights = [g.edges[chosen_node, neigh]["weight"] for neigh in neighbors]
        weights = [x/sum(weights) for x in weights]  # normalize
        chosen_neighbor_i = np.random.choice(range(len(neighbors)), p=weights)
        chosen_neighbor = neighbors[chosen_neighbor_i]
        g.nodes[chosen_neighbor]["wordform"] = g.nodes[chosen_node]["wordform"]  # perform the extension
    else:
        # replace
        existing_wordforms = [g.nodes[node]["wordform"] for node in g.nodes]
        while True:
            new_wf = get_wordform()
            if new_wf not in existing_wordforms:
                break
        g.nodes[chosen_node]["wordform"] = new_wf

    return g


def make_n_semantic_changes(g, n):
    for i in range(n):
        g = make_semantic_change(g)
    return g


def print_form_meaning_correspondence(g):
    print("\n-- form-meaning correspondence")
    d = {}
    for n in g.nodes:
        wf = g.nodes[n]["wordform"]
        if wf not in d:
            d[wf] = []
        d[wf].append(n)
    for form in sorted(d.keys()):
        concepts = d[form]
        concepts_str = ", ".join(concepts)
        print("wordform '{}' codes {} concepts: {}".format(form, len(concepts), concepts_str))


def evolve_language_on_conceptual_space(conceptual_space):
    g0 = conceptual_space
    g1 = add_initial_forms(g0)
    g1 = make_n_semantic_changes(g1, 100)
    return g1


def draw_weighted_graph(g):
    # https://stackoverflow.com/questions/22967086/colouring-edges-by-weight-in-networkx
    edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())
    pos = nx.spring_layout(g)
    nx.draw(g, pos, node_color='b', edgelist=edges, edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.show()


def draw_semantic_map(language_specific_graphs):
    # for mid-edge text labels: https://stackoverflow.com/questions/47094949/labeling-edges-in-networkx
    pos = nx.spring_layout(g)
    nx.draw(G,pos,edge_color='black',width=1,linewidths=1,\
    node_size=500,node_color='pink',alpha=0.9,\
    labels={node:node for node in G.nodes()})
    nx.draw_networkx_edge_labels(G,pos,edge_labels={('A','B'):'AB',\
    ('B','C'):'BC',('B','D'):'BD'},font_color='red')
    plt.show()


if __name__ == "__main__":
    g0 = create_conceptual_space(50)
    # draw_weighted_graph(g)

    g1 = evolve_language_on_conceptual_space(g0)
    g2 = evolve_language_on_conceptual_space(g0)
    g3 = evolve_language_on_conceptual_space(g0)
    print_form_meaning_correspondence(g1)
    print_form_meaning_correspondence(g2)
    print_form_meaning_correspondence(g3)

    draw_semantic_map([g1, g2, g3])

