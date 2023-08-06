# make some visualizations of cluster structures

import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def get_points(dim):
    assert dim == 2 or dim == 3
    return get_points_by_graph_association(dim)


def get_points_by_graph_association(dim):
    # do something like: make a graph where each point has a random 5 neighbors and 5 enemies
    # then at each step, each point is drawn toward the average position of its neighbors and away from that of its enemies
    # iterate this and see what happens
    # if it doesn't look good enough, try changing the friends and enemies sometimes
    assert dim == 2 or dim == 3

    n = 250
    n_steps = 500
    friends = {i: None for i in range(n)}
    enemies = {i: None for i in range(n)}
    positions = {i: np.random.normal(0, 1, (dim,)) for i in range(n)}
    for i in range(n):
        n_associates = random.randint(2, n-1)
        associates = random.sample([x for x in range(n) if x != i], n_associates)
        friend_ratio = random.uniform(1/n_associates, 1 - 1/n_associates)
        n_friends = int(round(friend_ratio * n_associates))
        n_enemies = n_associates - n_friends
        assert 1 <= n_friends
        assert 1 <= n_enemies
        friends[i] = associates[:n_friends]
        enemies[i] = associates[n_friends:]
    for step_i in range(n_steps):
        new_positions = {}
        for i in range(n):
            my_friends = friends[i]
            my_enemies = enemies[i]
            friend_avg_pos = 1/len(my_friends) * sum(positions[j] for j in my_friends)
            enemy_avg_pos = 1/len(my_enemies) * sum(positions[j] for j in my_enemies)
            r_friend = friend_avg_pos - positions[i]
            r_enemy = enemy_avg_pos - positions[i]
            mag_friend = min(1, 1/(np.linalg.norm(r_friend))**2)
            mag_enemy = min(1, 1/(np.linalg.norm(r_enemy))**2)
            d_pos = mag_friend * r_friend - mag_enemy * r_enemy
            new_positions[i] = positions[i] + d_pos
        positions = new_positions
    return np.array([positions[i] for i in range(n)]).T


if __name__ == "__main__":
    dim = 2

    if dim == 2:
        xs, ys = get_points(dim)
        plt.scatter(xs, ys)
        ax = plt.gca()
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        xs, ys, zs = get_points(dim)
        ax.scatter(xs, ys, zs)
    else:
        raise ValueError(dim)

    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

