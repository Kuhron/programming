# simulate fire spreading

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from scipy.signal import convolve2d

sys.path.insert(0, "/home/wesley/programming/")
from Emergence.PerlinNoise import get_2d_noise
from InteractivePlot import InteractivePlot



def get_dryness_array(size, resolution):
    arr = get_2d_noise(min_val=0, max_val=1, x_range=size, y_range=size, n_xs=resolution, n_ys=resolution)
    p = random.uniform(1, 2.5)
    return arr ** p  # lower the dryness values for wetter areas


def get_fuel_array(size, resolution):
    arr = get_2d_noise(min_val=0, max_val=100, x_range=size, y_range=size, n_xs=resolution, n_ys=resolution)
    return arr


def get_initial_fire_array(resolution):
    spark_location = (random.randrange(dryness.shape[0]), random.randrange(dryness.shape[1]))
    arr = np.zeros(shape=(resolution, resolution))
    initial_intensity = random.random()
    x, y = spark_location
    arr[x, y] = initial_intensity
    return arr


def evolve(dryness, fuel, fire):
    # suppose fire spreads to D4 neighbors
    # influence on an already-burning area should intensify the fire
    # influence on a burned-out area should do nothing (fuel=0)
    # influence on a to-burn area will have some chance of starting a blaze depending on the dryness
    # intensity (value of fire array) is the same as rate of fuel depletion
    # so steps:
    # 1. fire gives influence to neighbors
    # 2. start new fires
    # 3. add intensity to existing fires
    # 4. deplete fuel
    # 5. put out fire if fuel is gone

    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
    influence = convolve2d(fire, kernel, mode="same")
    influence_as_probability = 1 - 1/(1 + influence)
    assert (0 <= influence_as_probability).all() and (influence_as_probability <= 1).all()
    burned_out = fuel == 0
    burning = fire > 0
    can_start_fire = (fuel > 0) & (fire == 0)

    probability_of_starting_fire = influence_as_probability * dryness * can_start_fire.astype(int)
    r = np.random.random(fire.shape)
    places_with_new_fires = r < probability_of_starting_fire
    new_fire_intensity = r * places_with_new_fires.astype(int)
    fire += new_fire_intensity

    existing_fire_intensification = influence * burning.astype(int)
    fire += existing_fire_intensification

    # deplete intensity by heat loss to air
    fire -= 0.01
    fire = np.maximum(0, fire)

    fuel -= fire
    fuel = np.maximum(0, fuel)
    fire[fuel == 0] = 0

    assert (fuel >= 0).all()
    assert (fire >= 0).all()
    return dryness, fuel, fire


def plot_fire(dryness, fuel, fire, plt=plt):
    plt.subplot(1, 3, 1)
    cmap = cm.get_cmap("jet")
    plt.imshow(dryness, cmap=cmap)
    plt.colorbar()
    plt.title("dryness")

    plt.subplot(1, 3, 2)
    cmap = cm.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="black")
    cmap.set_under(color="black")
    plt.imshow(fuel, cmap=cmap, vmin=1e-6)
    plt.colorbar()
    plt.title("fuel")

    plt.subplot(1, 3, 3)
    cmap = cm.get_cmap("autumn").copy()
    cmap.set_bad(color="black")
    cmap.set_under(color="black")
    plt.imshow(fire, cmap=cmap, vmin=1e-6)
    plt.colorbar()  # jumps around too much when you put the frames together into a video
    plt.title("fire intensity")


if __name__ == "__main__":
    size = 4
    resolution = 150
    plot_every_n_steps = 1

    dryness = get_dryness_array(size, resolution)
    fuel = get_fuel_array(size, resolution)
    fire = get_initial_fire_array(resolution)

    # a location can be not on fire, on fire with a certain intensity, or burned out
    # a burned-out location won't re-ignite
    # a location that is on fire can be intensified/prolonged by neighboring fire
    # a location that has not been on fire yet is likely to receive fire from its neighbors based on their combined intensity and its own dryness
    # so an intense fire will deplete fuel faster
    # I guess fire intensity is unbounded, if there's enough fuel in that cell it can keep going up

    with InteractivePlot(plot_every_n_steps, suppress_show=True, figsize=(12,3)) as iplt:
        while iplt.is_open():
            plot_fire(dryness, fuel, fire, plt=iplt)
            dryness, fuel, fire = evolve(dryness, fuel, fire)
            iplt.step(savefig=True)

            if (fire == 0).all():
                plot_fire(dryness, fuel, fire, plt=iplt)
                iplt.force_draw_static(savefig=False)  # even if it's not the right counter number in the iplt
                input("the fire is over; press enter to close")
                break

