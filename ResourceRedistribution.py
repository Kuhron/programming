# from high idea on 2025-03-28:
# random fluctuations create temporary inequality
# how does the social network respond to this?
# if they are generous in sharing from those who have too much to those who have too little, things will rebalance
# but there may also be a cognitive self-protection mechanism by which those who have had a brush with death (low resources) develop protectionist walls (becoming less likely to share and more likely to hoard)
# depending on the functional form of this wall's permeability as a function of time since near-death experience and severity of the experience, how will the network react macroscopically?
# suspect that the more people do this self-protecting behavior, the more the network will have pockets of overdensity and underdensity, creating persistent inequality
# simulate it and see what happens in different conditions
# start with everyone having the same wall function, then try where different people have different "personalities" in terms of this function
# another variable: what information does each person have about the needs of others? if they only know about their neighbors and give accordingly, might still have some inefficiencies, try incorporating a mechanism about how much they know about all the other people's needs and how that influences their choices of which of their own neighbors to send resources to (hoping that those neighbors will in turn send those resources onward to those who need them most, or the resources can be earmarked in a package that the intermediaries are not allowed to open, the package label containing directions for where it is to go each step until it reaches its intended recipient), experiment with simulating the resource redistribution under each of these possibilities


import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import CirclePolygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize



# assume simple graph: hexagonal lattice on toroidal array, so we can easily plot it and get neighbors of each point

# enforce number of rows and columns being even, where this is 4 rows by 10 columns:
#   x x x x x
#  x x x x x
#   x x x x x
#  x x x x x
# (origin in lower left)

R32 = 1/2 * (3**0.5)


def rc_to_xy(rc):
    a = 1  # side length on the lattice
    x_offset_for_odd_row = 1/2 * a
    dx_per_column = a
    dy_per_row = a * R32
    r,c = rc
    x = (c * dx_per_column) + (r % 2 == 1) * x_offset_for_odd_row
    y = r * dy_per_row
    return x,y


def plot_grid(X, Y, vals, walls, val_cmap, wall_cmap, wall_width_func, show=False):
    # TODO make wall thickness reflected in thickness and/or color of a border around the point marker
    fig, ax = plt.subplots()

    # val_mappable = plt.scatter(X, Y, c=vals, cmap=val_cmap, alpha=0)
    val_mappable = ScalarMappable(Normalize(np.min(vals), np.max(vals)), cmap=val_cmap)
    wall_mappable = ScalarMappable(Normalize(np.min(walls), np.max(walls)), cmap=wall_cmap)

    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            x = X[r,c]
            y = Y[r,c]
            val = vals[r,c]
            wall = walls[r,c]
            facecolor = val_mappable.to_rgba(val)
            edgecolor = wall_mappable.to_rgba(wall)
            edgewidth = wall_width_func(wall)
            circle = CirclePolygon((x,y), radius=0.35, facecolor=facecolor, edgecolor=edgecolor, linewidth=edgewidth)
            ax.add_patch(circle)

    ax.set_aspect("equal")

    ax.set_xlim(np.min(X)-1, np.max(X)+1)
    ax.set_ylim(np.min(Y)-1, np.max(Y)+1)

    val_bar = plt.colorbar(val_mappable, ax=ax)
    # wall_bar = plt.colorbar(wall_mappable, ax=ax)

    if show:
        plt.show()



if __name__ == "__main__":
    n_rows = 20
    n_cols = 20

    R,C = np.meshgrid(range(n_rows), range(n_cols))
    X,Y = rc_to_xy((R,C))

    vals = np.random.normal(100, 1, X.shape)
    walls = np.random.random(X.shape)

    val_cmap = plt.get_cmap("jet")
    wall_cmap = plt.get_cmap("binary")
    wall_width_func = lambda x: x

    # TODO watch evolution by using the interactive plot context manager I wrote a while back
    plot_grid(X, Y, vals, walls, val_cmap, wall_cmap, wall_width_func, show=False)
    plt.show()

