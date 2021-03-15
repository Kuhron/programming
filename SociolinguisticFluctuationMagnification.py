# simulation of the idea that pre-existing social categories will magnify differences in language
# do this by making some agents, each with a set of social variables
# each individual also has a diversity tolerance level, which determines how much they will socialize with people with different social variables
# make a social network (graph) among them, based on who they socialize with
# 
# once the whole network and social structure exists, give everyone a "vocabulary" (array of strings to represent the language)
# add random fluctuations without regard to the social network
# people are more likely to copy those like them and try not to be like those not like them (also conditional on their diversity tolerance)

import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools

from WordGenerationBasic import get_wordform


class SocialAgent:
    def __init__(self, observable_variables, diversity_tolerance, sociability):
        self.observable_variables = observable_variables
        self.vector = np.array([v for k,v in sorted(observable_variables.items())])
        self.diversity_tolerance = diversity_tolerance
        self.sociability = sociability

    def get_social_distance(self, other):
        return np.linalg.norm(self.vector - other.vector) / len(self.vector)  # normalize to [0,1] regardless of number of variables

    def will_socialize(self, other):
        return (random.random() < self.sociability) and (self.get_social_distance(other) <= self.diversity_tolerance)



def create_social_agents(n_agents):
    # let all the variables be continuous so they get a gradient distance from their compatriots
    social_variables = ["gender", "age", "race", "income"]
    # treat all as independent for first pass
    # don't measure people's distances in diversity-tolerance space, don't let them be able to observe that value of others
    agents = []
    for i in range(n_agents):
        d = {var: random.random() for var in social_variables}
        diversity_tolerance = random.uniform(0, 0.3)
        sociability = random.random()
        agent = SocialAgent(d, diversity_tolerance, sociability)
        agents.append(agent)
    return agents


def create_social_network(agents):
    g = nx.Graph()

    # look at the potential edges in random order, because I am giving people a maximum social load, so early numbers won't be favored as friends because they came up first
    combos = list(itertools.combinations(range(len(agents)),2))
    random.shuffle(combos)  # I hate that this function is in-place

    for i in range(len(agents)):
        g.add_node(i)

    for i,j in combos:
        ai = agents[i]
        aj = agents[j]
        i_socialize_j = ai.will_socialize(aj)
        j_socialize_i = aj.will_socialize(ai)
        if i_socialize_j and j_socialize_i:
            g.add_edge(i,j)

    # force loners to have at least one friend, and force the whole network to be connected
    while not nx.is_connected(g):
        components = list(nx.connected_components(g))
        c0, c1 = random.sample(components, 2)  # these are sets of node names
        n0 = random.choice(list(c0))
        n1 = random.choice(list(c1))
        g.add_edge(n0, n1)
    return g


def create_base_vocabulary(n_lexemes=100):
    res = set()
    while len(res) < n_lexemes:
        w = get_wordform()
        res.add(w)
    return sorted(res)



if __name__ == "__main__":
    agents = create_social_agents(n_agents=100)
    network = create_social_network(agents)

    for i in network.nodes:
        print("person {} has {} friends".format(i, network.degree(i)))
    nx.draw(network, with_labels=True)
    plt.show()

    base_vocabulary = create_base_vocabulary(n_lexemes=100)


