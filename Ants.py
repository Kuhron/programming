# try to simulate ants looking for food and then following pheromone trails

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import convolve2d
import random

from InteractivePlot import InteractivePlot


class Ant:
    def __init__(self, location):
        self.location = location
        self.food_knowledge = False


def get_random_location(grid_shape):
    nr, nc = grid_shape
    return (random.randrange(nr), random.randrange(nc))


def get_food_grid(grid_shape):
    grid = np.zeros(grid_shape)
    n_cells = grid.size
    freq = 1/1000
    n_foods = max(1, round(freq*n_cells))
    for i in range(n_foods):
        location = get_random_location(grid_shape)
        grid[location] = random.random() * 10
    # convolve to make them like piles of food
    kernel = np.array([[0.3, 0.7, 0.3], [0.7, 1, 0.7], [0.3, 0.7, 0.3]])
    for i in range(3):
        grid = convolve2d(grid, kernel, mode="same")
    return grid


def diffuse_pheromones(grid):
    a = 0.05
    kernel = np.array([[0, a, 0], [a, 1-6*a, a], [0, a, 0]])  # spread them and also decay the total
    return convolve2d(grid, kernel, mode="same")


def add_food_pheromones(ants, food_grid, food_pheromone_grid):
    for ant in ants:
        loc = ant.location
        food = food_grid[loc]
        food_pheromone_grid[loc] += ant.food_knowledge
    return food_pheromone_grid


def add_history_pheromones(ants, history_pheromone_grid):
    for ant in ants:
        loc = ant.location
        history_pheromone_grid[loc] += 1
    return history_pheromone_grid


def move_ants(ants, anthill_location, food_grid, food_pheromone_grid, history_pheromone_grid):
    new_ants = []
    grid_shape = food_grid.shape
    for ant in ants:
        loc = ant.location
        food_here = food_grid[loc]
        if food_here > 0:
            ant.food_knowledge = food_here * 10
        else:
            ant.food_knowledge *= 0.95  # exponential decay
        neighbors = get_d8_neighbors(loc, grid_shape)
        # choose which neighbor to go to based on combination of food, food pheromones, and history pheromones
        scores = []
        for nloc in neighbors:
            food_pheromone_at_neighbor = food_pheromone_grid[nloc]
            history_pheromone_at_neighbor = history_pheromone_grid[nloc]

            # if there's food that you know about, prioritize going back to tell others
            food_communication_score = ant.food_knowledge * history_pheromone_at_neighbor
            # if there's no food and you don't know about food, prioritize following food pheromones and avoiding history
            following_score = food_pheromone_at_neighbor - history_pheromone_at_neighbor

            score = 10 * food_communication_score + following_score
            score = max(0, score)
            scores.append(score)
        # go probabilistically based on weighted average of scores
        if sum(scores) == 0:
            new_loc = random.choice(neighbors)
        else:
            scores = [x/sum(scores) for x in scores]
            new_loc = neighbors[np.random.choice(list(range(8)), p=scores)]
        ant.location = new_loc
        if new_loc == anthill_location:
            pass  # delete the ant, it goes back home / forgets about food
        else:
            new_ants.append(ant)
    return new_ants


def get_d8_neighbors(loc, grid_shape):
    # do toroidal array because it's easier
    r, c = loc
    nr, nc = grid_shape
    return [((r+i) % nr, (c+j) % nc) for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0]


def eat_food(ants, food_grid):
    for ant in ants:
        loc = ant.location
        food = food_grid[loc]
        food = max(0, food - 1)
        food_grid[loc] = food
    return food_grid


def plot_situation(ants, food_grid, food_pheromone_grid, history_pheromone_grid, plt):
    nr, nc = food_grid.shape
    plt.subplot(2,2,1)
    plt.gca().set_xlim((0, nr))
    plt.gca().set_ylim((0, nc))
    plt.gca().set_aspect("equal")
    for ant in ants:
        r, c = ant.location
        x, y = c, r
        plt.scatter(x, y, c="k")
    plt.title("ants")

    cmap = cm.get_cmap("RdYlGn").copy()
    # cmap.set_under(color="black")
    # vmin = 0

    plt.subplot(2,2,2)
    plt.imshow(food_grid, origin="lower", cmap=cmap)
    plt.title("food")

    plt.subplot(2,2,3)
    plt.imshow(history_pheromone_grid, origin="lower", cmap=cmap)
    plt.title("history pheromones")

    plt.subplot(2,2,4)
    plt.imshow(food_pheromone_grid, origin="lower", cmap=cmap)
    plt.title("food_pheromones")




if __name__ == "__main__":
    grid_shape = (100, 100)
    food_grid = get_food_grid(grid_shape)
    food_pheromone_grid = np.zeros(grid_shape)  # ants follow food pheromone to get food
    history_pheromone_grid = np.zeros(grid_shape)  # ants follow history pheromone to retrace their steps
    anthill_location = (grid_shape[0]//2, grid_shape[1]//2)
    ants = [Ant(anthill_location)]
    max_ants = 250

    with InteractivePlot() as iplt:
        t = 0
        while True:
            print(f"t = {t}, {len(ants)} ants are out")
            if random.random() < 0.1 and len(ants) < max_ants:
                # emit one ant
                ant = Ant(anthill_location)
                ants.append(ant)
            ants = move_ants(ants, anthill_location, food_grid, food_pheromone_grid, history_pheromone_grid)
            food_grid = eat_food(ants, food_grid)
            food_pheromone_grid = add_food_pheromones(ants, food_grid, food_pheromone_grid)
            history_pheromone_grid = add_history_pheromones(ants, history_pheromone_grid)
            food_pheromone_grid = diffuse_pheromones(food_pheromone_grid)
            history_pheromone_grid = diffuse_pheromones(history_pheromone_grid)

            plot_situation(ants, food_grid, food_pheromone_grid, history_pheromone_grid, plt=iplt)
            iplt.draw()
            t += 1
