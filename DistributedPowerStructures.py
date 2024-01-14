# TODO idea: instead of tree-like structure of government,
# try more distributed structure where in each of many domains there is a separate tree-ish structure leading to a panel of experts (but then can redo the tree-to-distributed transformation on those if necessary to make it even less power-pyramid-like)
# find a way to simulate decision making in such a system, see what they do versus what a tree-like power structure would do

# TODO use networkx to make directed graph with weighted edges showing power relationships (can do this each of for multiple kinds of power among individuals in a group)
# - visualize it such that a hierarchical-like power structure ends up shaped like a tree graph (at least when looking at the strongest edges, and other weak edges may crisscross the diagram in various ways, but when you get the gist it looks like a tree)
# - simulate some dynamics so that the network changes regimes, such as becoming more like a distributed network / polygon with various edges crossing in various directions, rather than a hierarchical tree structure
# - animate the network's evolution

# TODO make sound from the dynamics somehow, so I can hear stuff like pulsations as the edge weights change, see what a regime change sounds like


import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


