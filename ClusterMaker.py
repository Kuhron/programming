# make some visualizations of cluster structures

import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix


def get_points(dim):
    assert dim == 2 or dim == 3
    return get_points_by_association(dim)


def get_points_by_association(dim):
    # do something like: make a graph where each point has a random 5 neighbors and 5 enemies
    # then at each step, each point is drawn toward the average position of its neighbors and away from that of its enemies
    # iterate this and see what happens
    # if it doesn't look good enough, try changing the friends and enemies sometimes

    # another idea: make the points more or less like each other based on some arbitrary attributes, base their attraction on that
    assert dim == 2 or dim == 3

    n = 1000
    n_steps = 70

    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
    attributes = np.random.uniform(-1, 1, (n, 4))  # do fewer dimensions if points are still too centrally clustered
    attributes = mod_unit_sphere(attributes)
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(attributes)
    _, indices = nbrs.kneighbors(attributes)
    for i in range(n):
        neighbor_indices = indices[i][1:]
        assert i not in neighbor_indices
        for j in neighbor_indices:
            g.add_edge(i, j)

    positions = np.random.normal(0, 1, (n, dim))
    for step_i in range(n_steps):
        if step_i % 10 == 0:
            print(f"step {step_i}/{n_steps}")
        distances = distance_matrix(positions, positions)  # will change
        nbrs = NearestNeighbors(n_neighbors=18, algorithm='ball_tree').fit(positions)  # include some gravity from actual spatial neighbors as well
        _, nbr_indices = nbrs.kneighbors(positions)
        new_positions = np.zeros((n, dim))
        for i in range(n):
            neighbors_index = np.array(list(set(x for x in g.neighbors(i)) | set(j for j in nbr_indices[i] if j != i)))
            displacements = positions[neighbors_index] - positions[i]  # needed for what direction we'll move in
            mags = (lambda r: 1/(1e-9+r)**2)(distances[i, neighbors_index])
            # forces = (lambda w: -1 + w*(1 - -1))(similarities[i, not_self_index]) * mags
            # forces = np.vectorize(lambda w: -0.1 if w < 0.4 else 0.1 if w > 0.6 else 0)(similarities[i, not_self_index]) * mags
            forces = np.full((len(neighbors_index), 1), 0.1)
            d_pos = (displacements * forces).sum(axis=0)
            new_pos = positions[i] + d_pos
            new_positions[i] = new_pos
        new_positions = mod_unit_sphere(new_positions)
        positions = new_positions
    return positions.T


def mod_unit_sphere(pos):
    # take the displacement vector from the origin to the point, mod that such that it wraps around on the same line inside the unit sphere
    r = np.linalg.norm(pos, axis=1).reshape(pos.shape[0], 1)
    normals = pos/r
    n_to_subtract = 2 * ((r + 1) // 2)  # if r < 1, do nothing; if 1 <= r < 3, subtract 2 normal vecs; if 3 <= r < 5, subtract 4; etc.
    new_pos = pos - n_to_subtract * normals
    assert (np.linalg.norm(new_pos, axis=1) < 1).all()
    return new_pos


if __name__ == "__main__":
    dim = 2
    mds = False
    axis = False

    if dim == 2:
        xs, ys = get_points(dim)
        plt.scatter(xs, ys)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        coords = np.array([xs, ys]).T
    elif dim == 3:
        xs, ys, zs = get_points(dim)

        if mds:
            xs, ys = MDS().fit_transform(np.array([xs, ys, zs]).T).T
            plt.scatter(xs, ys)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            coords = np.array([xs, ys]).T
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(xs, ys, zs)
            coords = np.array([xs, ys, zs]).T
    else:
        raise ValueError(dim)

    if not axis:
        ax.set_axis_off()
    plt.show()

    if input("save csv? (y for yes)") == "y":
        with open("cluster_data.csv", "w") as f:
            for row in coords:
                s = ",".join(str(x) for x in row) + "\n"
                f.write(s)

