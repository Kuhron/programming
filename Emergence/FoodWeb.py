# trying to simulate a basic food web or similar ecological system, see how it weathers shocks and/or undergoes regime shift

import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt


class Species:
    def __init__(self, name):
        self.name = name
        self._next_id_num = 0  # don't access directly from outside the class (private, but Python doesn't have that)

    def __repr__(self):
        return self.name

    def get_next_id_num(self):
        n = self._next_id_num
        self._next_id_num += 1
        return n


class Organism:
    def __init__(self, species, energy_need, autotrophic_production, heterotrophic_efficiency, fertility_rate, energy_stored, location, biome):
        self.species = species
        self.id_num = self.species.get_next_id_num()
        self.e = energy_need
        self.a = autotrophic_production
        self.h = heterotrophic_efficiency
        self.r = fertility_rate
        self.energy_stored = energy_stored
        self.location = location
        self.biome = biome
        self.biome.location_to_organisms[location].append(self)
        self.alive = True
        # print(f"created {self}")

    def __repr__(self):
        return f"<{'*' if not self.alive else ''}{self.species} {self.id_num} e={self.e:.4f} a={self.a:.4f} h={self.h:.4f} r={self.r:.4f} @ {self.location}>"

    def to_short_str(self):
        return f"<{'*' if not self.alive else ''}{self.species} {self.id_num}>"

    def get_food(self):
        if not self.alive:
            # print(f"can't get food if you're dead: {self.to_short_str()}")
            return

        # print(f"getting food for {self}")
        # self.biome.print_locations()
        assert self in self.biome.location_to_organisms[self.location], self

        # first try making your own food, see if that's enough
        p = 0.5
        a = self.a * (2**random.uniform(-p, p))  # how much energy we produce this time
        e = self.e * (2**random.uniform(-p, p))  # how much energy we need this time
        # print(f"need {e:.4f}, produced {a:.4f}")
        if a >= e:
            self.energy_stored += (a - e)
            # print(f"consumed autotrophically: {e:.4f}, stored {a-e:.4f}, have {self.energy_stored:.4f} stored")
        else:
            e -= a  # use what we made, still need some
            # print(f"consumed autotrophically: {a:.4f}, still need {e:.4f}")
            e = self.hunt(e)
            if e > 0:
                e = self.starve(e)
            if e > 0:
                # we couldn't get enough energy from any source
                self.die()
        # print(f"done getting food for {self}")

    def hunt(self, e, can_move_again=True):
        # stay in this spot if it will work okay
        # allow cannibalism? sure why not, real animals do it, no auto-cannibalism though, that's what starve() is for
        if not self.alive:
            return None
        cands = [o for o in self.biome.location_to_organisms[self.location] if o is not self]
        # print(f"found {len(cands)} organisms for potential eating")
        for o in cands:
            if self.will_eat(o):
                amount_to_consume = o.energy_stored
                if amount_to_consume > e:
                    # consume what we need and store the rest
                    e = 0
                    self.energy_stored += amount_to_consume - e
                else:
                    e -= amount_to_consume
                o.die()
                # print(f"{self.to_short_str()} ate {o.to_short_str()}, consumed {amount_to_consume:.4f}, still need {e:.4f}")
                if e <= 0:
                    break
        # print(f"done hunting at location {self.location}")

        if e < 0:
            raise RuntimeError
        elif e == 0:
            return e  # finished hunting

        if can_move_again:
            # if run out of food here, move to a neighboring spot (expend stored energy to do so)
            # print("moving to find more food")
            new_location = self.biome.get_neighboring_location(self.location)
            self.move(new_location)
            return self.hunt(e, can_move_again=False)
        else:
            # print("staying in place")
            return e

    def move(self, new_location):
        self.biome.move_organism(self, new_location)
        self.energy_stored *= 0.75

    def starve(self, e):
        # consume stored energy
        amount_can_consume = min(e, self.energy_stored)
        self.energy_stored -= amount_can_consume
        e -= amount_can_consume
        # print(f"consumed stored energy: {amount_can_consume:.4f}, still need {e:.4f}, have {self.energy_stored:.4f} stored")
        return e

    def die(self):
        self.alive = False
        self.biome.location_to_organisms[self.location].remove(self)  # although it could stay there and be eaten by a scavenger
        # print(f"died: {self}")

    def reproduce(self):
        if not self.alive:
            return
        # look for a mate in current spot
        cands = [o for o in self.biome.location_to_organisms[self.location] if o.species is self.species and o is not self]
        # print(f"found {len(cands)} conspecifics for reproduction")
        random.shuffle(cands)
        offspring = []
        for o in cands:
            if self.will_reproduce(o):
                offspring = self.make_offspring(o)
                break
        if offspring == []:
            # print("did not reproduce")
            pass
        return offspring

    def make_offspring(self, o):
        s = self.species
        assert o.species is s
        assert o.location == self.location
        assert o.biome is self.biome
        e = (self.e * o.e) ** 0.5
        a = (self.a * o.a) ** 0.5
        h = (self.h * o.h) ** 0.5
        r = (self.r * o.r) ** 0.5
        offspring = []
        n_offspring_to_make = int(round(r))
        if n_offspring_to_make > 0:
            energy_to_give_to_each_offspring = (1/2 * (self.energy_stored) + 1/2 * (o.energy_stored)) / n_offspring_to_make
            for i in range(n_offspring_to_make):
                p = 0.1
                e1 = e * (2**random.uniform(-p, p))
                a1 = a * (2**random.uniform(-p, p))
                h1 = h * (2**random.uniform(-p, p))
                r1 = r * (2**random.uniform(-p, p))
                g1 = energy_to_give_to_each_offspring
                new_o = Organism(species=s, energy_need=e1, autotrophic_production=a1, heterotrophic_efficiency=h1, fertility_rate=r1, energy_stored=g1, location=self.location, biome=self.biome)
                offspring.append(new_o)
            self.energy_stored /= 2  # given to offspring
            o.energy_stored /= 2  # given to offspring
        # print(f"{self} reproduced with {o} to create {n_offspring_to_make} offspring")
        return offspring

    def will_eat(self, other):
        probability = 1  # make this more complicated later
        return random.random() < probability

    def will_reproduce(self, other):
        probability = 1  # make this more complicated later
        return random.random() < probability


class Biome:
    def __init__(self, side_length):
        self.side_length = side_length
        self.location_to_organisms = {(i,j): [] for i in range(side_length) for j in range(side_length)}

    def move_organism(self, o, new_location):
        # print(f"moving {o} to {new_location}")
        self.location_to_organisms[o.location].remove(o)
        self.location_to_organisms[new_location].append(o)
        o.location = new_location
        # print(f"done moving {o} to {new_location}")

    def get_random_location(self):
        return (random.randrange(self.side_length), random.randrange(self.side_length))

    def get_neighboring_location(self, location):
        x,y = location
        dxs = [1] if x == 0 else [-1] if x == (self.side_length - 1) else [1, -1]
        dys = [1] if y == 0 else [-1] if y == (self.side_length - 1) else [1, -1]
        tups = [(dx, 0) for dx in dxs] + [(0, dy) for dy in dys]
        dx, dy = random.choice(tups)
        new_x, new_y = x+dx, y+dy
        new_loc = (new_x, new_y)
        assert self.contains(new_loc)
        return new_loc

    def contains(self, location):
        x, y = location
        return 0 <= x < self.side_length and 0 <= y < self.side_length

    def print_locations(self):
        print()
        for loc in sorted(self.location_to_organisms.keys()):
            print(f"{loc} : {', '.join(o.to_short_str() for o in self.location_to_organisms[loc])}")
        print()


def create_organisms(species, initial_populations, energy_needs, autotrophic_productions, heterotrophic_efficiencies, fertility_rates, biome):
    organisms = []
    for s_i in range(len(species)):
        s = species[s_i]
        n = initial_populations[s_i]
        e = energy_needs[s_i]
        a = autotrophic_productions[s_i]
        h = heterotrophic_efficiencies[s_i]
        r = fertility_rates[s_i]
        for i in range(n):
            p = 0.1
            e1 = e * (2**random.uniform(-p, p))
            a1 = a * (2**random.uniform(-p, p))
            h1 = h * (2**random.uniform(-p, p))
            r1 = r * (2**random.uniform(-p, p))
            g1 = e1 * 2.5
            location = biome.get_random_location()
            organism = Organism(species=s, energy_need=e1, autotrophic_production=a1, heterotrophic_efficiency=h1, fertility_rate=r1, energy_stored=g1, location=location, biome=biome)
            organisms.append(organism)
    return organisms


if __name__ == "__main__":
    s1 = Species("cazhok")
    s2 = Species("motup")
    s3 = Species("ambras")
    s4 = Species("orsil")
    s5 = Species("laprentar")
    s6 = Species("darabom")
    s7 = Species("grattoon")
    species = [s1, s2, s3, s4, s5, s6, s7]

    # seed = input("seed: ")
    seed = time.time() + (time.time() % 1)*1e10
    try:
        seed = int(seed)
        random.seed(seed)
        # print(f"{seed = }")
    except ValueError:
        pass

    biome = Biome(10)

    initial_populations = [random.randint(20, 100) for s in species]
    energy_needs = [random.uniform(2, 20) for s in species]
    autotrophic_productions = [random.uniform(2, 20) for s in species]
    heterotrophic_efficiencies = [random.random() for s in species]
    fertility_rates = [random.uniform(0, 6) for s in species]

    organisms = create_organisms(species, initial_populations, energy_needs, autotrophic_productions, heterotrophic_efficiencies, fertility_rates, biome)
    # print()

    population_time_series = {s.name: [sum(o.species is s for o in organisms)] for s in species}
    for i in range(100):
        print(f"generation {i}")
        all_offspring = []
        random.shuffle(organisms)
        for o_i, o in enumerate(organisms):
            print(f"{o_i}/{len(organisms)}", end="\r")
            if not o.alive:
                continue
            o.get_food()
            if o.alive:
                offspring = o.reproduce()
                all_offspring += offspring

        # now remove dead ones so don't get list changed during iteration
        organisms += all_offspring
        organisms = [o for o in organisms if o.alive]

        for s in species:
            population_time_series[s.name].append(sum(o.species is s for o in organisms))

        # input("press enter to continue")
        # print()

    for s in species:
        plt.plot(population_time_series[s.name], label=s.name)
    plt.legend()
    plt.show()
