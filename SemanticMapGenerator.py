import random
import string
import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def get_concept_names_cmudict(n_concepts):
    with open("cmudict.txt") as f:
        lines = f.readlines()
    names = list(set(l.split()[0] for l in lines))  # some words have more than one line in the cmudict, filter duplicates
    # filter out some crappy ones, e.g. all the ones which are possessive
    names = [x for x in names if not x.endswith("'S") and not x.endswith("'")]
    return random.sample(names, n_concepts)


def get_concept_names_abc(n_concepts):
    gen = uppercase_abc_digit_generator()
    return [next(gen) for i in range(n_concepts)]


def uppercase_abc_digit_generator():
    letters = string.ascii_uppercase
    b = len(letters)
    gen = base_digit_list_generator_0_to_b_minus_1(b)
    for digits in gen:
        yield "".join(letters[i] for i in digits)


def base_digit_list_generator_0_to_b_minus_1(base):
    b = base
    digits = [0]
    while True:
        yield digits
        digits[-1] += 1
        # carry place values if necessary
        for dig_i in range(len(digits)):
            position = -(dig_i+1)
            dig = digits[position]
            if dig > b:
                raise Exception("shouldn't happen with base {}: {}".format(b, digits))
            elif dig == b:
                # carry to next position
                digits[position] = 0
                if dig_i == len(digits) - 1:
                    # we are already at most significant digit, need to create new one
                    digits = [0] + digits
                else:
                    digits[position-1] += 1
            else:
                # won't need to carry any more
                break


def create_conceptual_space(n_concepts):
    concepts = get_concept_names_abc(n_concepts)
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
    d = get_concepts_by_wordform_dict(g)
    for form in sorted(d.keys()):
        concepts = d[form]
        concepts_str = ", ".join(concepts)
        print("wordform '{}' codes {} concepts: {}".format(form, len(concepts), concepts_str))


def get_concepts_by_wordform_dict(language_specific_graph):
    g = language_specific_graph
    d = {}
    for n in g.nodes:
        wf = g.nodes[n]["wordform"]
        if wf not in d:
            d[wf] = []
        d[wf].append(n)
    return d


def evolve_language_on_conceptual_space(conceptual_space):
    g0 = conceptual_space
    g1 = add_initial_forms(g0)
    g1 = make_n_semantic_changes(g1, 100)
    return g1


def create_semantic_map(conceptual_space, language_specific_graphs):
    concepts = set(conceptual_space.nodes)
    assert all(set(g.nodes) == concepts for g in language_specific_graphs)

    # start from the combination of each gram's maximal graph, so we don't start with any edges which are definitely not needed
    # (i.e., we don't need every edge in the complete graph of the whole space)
    gram_specific_graphs = get_gram_complete_subgraphs(language_specific_graphs)
    maximal_graph = create_maximal_graph(concepts, gram_specific_graphs)

    # now check edges randomly for removability
    bridges = set()  # put non-removable edges here when they are found
    edges = set(maximal_graph.edges)
    g = maximal_graph
    while len(edges) > 0:
        e = random.choice(list(edges))
        u,v = e
        # remove the edge
        g.remove_edge(u,v)
        can_remove_edge = all_subgraphs_are_connected(g, gram_specific_graphs)
        if can_remove_edge:
            # leave g as it is, remove this edge from further consideration
            edges -= {e}
        else:
            # add the edge back
            g.add_edge(u,v)
            edges -= {e}
            bridges.add(e)
    return g


def all_subgraphs_are_connected(g, gram_specific_graphs):
    for gg in gram_specific_graphs:
        sg = g.subgraph(gg.nodes)
        if not nx.is_connected(g):
            return False
    return True


def create_maximal_graph(nodes, gram_specific_graphs):
    # add all edges from each gram-specific complete graph
    g = nx.Graph()
    g.add_nodes_from(nodes)  # populate each node, no edges yet
    for gg in gram_specific_graphs:
        for u,v in gg.edges:
            g.add_edge(u,v)
    return g


def get_gram_complete_subgraphs(language_specific_graphs):
    res = []
    for lg in language_specific_graphs:
        res += get_gram_complete_subgraphs_single_language(lg)
    return res


def get_gram_complete_subgraphs_single_language(language_specific_graph):
    # for each wordform in the language, make a complete graph, return all of these
    g = language_specific_graph
    res = []
    d = get_concepts_by_wordform_dict(g)
    for wf, concept_lst in d.items():
        subgraph = nx.generators.classic.complete_graph(concept_lst)
        res.append(subgraph)
    return res 


def simulate_languages(n_languages, n_concepts):
    g0 = create_conceptual_space(n_concepts)
    # draw_weighted_graph(g)

    language_graphs = []
    for i in range(n_languages):
        g1 = evolve_language_on_conceptual_space(g0)
        language_graphs.append(g1)
        print_form_meaning_correspondence(g1)

    draw_semantic_map(g0, language_graphs)


def draw_weighted_graph(g):
    # https://stackoverflow.com/questions/22967086/colouring-edges-by-weight-in-networkx
    edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())
    pos = nx.spring_layout(g)
    nx.draw(g, pos, node_color='b', edgelist=edges, edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.show()


def draw_semantic_map(conceptual_space, language_specific_graphs):
    # for mid-edge text labels: https://stackoverflow.com/questions/47094949/labeling-edges-in-networkx
    g = create_semantic_map(conceptual_space, language_specific_graphs)
    pos = nx.spring_layout(g)
    nx.draw(g, pos, 
        edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9, 
        labels={node:node for node in g.nodes()},
    )
    nx.draw_networkx_edge_labels(g, pos,
        edge_labels={(u,v): "{}-{}".format(u,v) for (u,v) in g.edges},
        font_color='red',
    )
    plt.show()



if __name__ == "__main__":
    simulate_languages(n_languages=5, n_concepts=100)
