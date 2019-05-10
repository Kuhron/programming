import math
import random
import numpy as np
import matplotlib.pyplot as plt


def distance(xy1, xy2):
    dx = xy2[0] - xy1[0]
    dy = xy2[1] - xy1[1]
    return (dx**2 + dy**2) ** 0.5

def average(xy1, xy2, alpha=0.5):
    x = alpha * xy1[0] + (1 - alpha) * xy2[0]
    y = alpha * xy1[1] + (1 - alpha) * xy2[1]
    return (x, y)


class Person:
    N_BASES = 50
    MINIMUM_REPRODUCTION_AGE = 3
    MIN_SCORE = float("-inf")

    def __init__(self, dna, land, location):
        self.dna = dna
        self.land = land
        self.location = location
        self.age = 0
        self.person_id = "{}-{}-{}".format(int(location[0]), int(location[1]), Person.value_gene_binary(self.dna))

        # genetic attributes
        self.similarity_dna = self.dna[:]
        self.similarity_weights = self.get_similarity_weights()
        self.genetic_age = Person.value_gene_binary(self.dna[5:11])
        self.neighbor_radius = 1 + Person.sum_gene(self.dna[2:11])
        self.max_neighbors = Person.sum_gene(self.dna[33:43])
        self.ideal_similarity = Person.value_gene_01(self.dna[9:24])
        self.reproduction_probability = Person.value_gene_01(self.dna[41:47])
        self.wanderlust_probability = Person.value_gene_01(self.dna[0:8])

    def __repr__(self):
        return "Person ID {}\ncoords {}\nage {}\nDNA {}\n".format(self.person_id, self.location, self.age, self.dna)

    def get_similarity_weights(self):
        s = self.similarity_dna
        raw_weights = []
        offset = 7
        determiner_length = 5
        for i in range(len(s)):
            determiner = "".join([s[(i + offset + j) % len(s)] for j in range(determiner_length)])
            raw_weight = Person.value_gene_01(determiner)
            raw_weights.append(raw_weight)
        total_weight = sum(raw_weights)
        if total_weight == 0:
            total_weight = 1
        return [w / total_weight for w in raw_weights]

    def select_mate(self, options):
        best_score = Person.MIN_SCORE
        best_option = None
        options = options[:]
        random.shuffle(options)
        for option in options:
            score = self.score(option)
            # print("score {}".format(score))
            if score > best_score:
                # print("BETTER")
                best_score = score
                best_option = option
        return best_option

    def score(self, other):
        if other.dna == self.dna:
            # no mating with yourself or your clone
            return Person.MIN_SCORE
        # select part of the dna for use as "similarity", not all of it
        d = Person.genetic_distance(self.similarity_dna, other.similarity_dna, weights=self.similarity_weights)
        s = self.ideal_similarity
        diff = abs(d - s)
        while diff == 0:
            # screw dealing with 1/0, just add some uncertainty so it's random among the perfect candidates
            diff = random.uniform(-1e-3, 1e-3)
        return 1/diff

    def reproduce(self, mate):
        new_dna = Person.combine_dna(self.dna, mate.dna)
        new_location = average(self.location, mate.location, alpha=random.random())
        return Person(new_dna, self.land, new_location)

    def maybe_reproduce(self):
        # decide whether to reproduce (based on something in DNA)
        offspring = None
        options = self.land.get_mating_options(self)
        will_reproduce = (not self.is_baby()) and self.check_dna_will_reproduce()
        if will_reproduce:
            # print("will reproduce")
            mate = self.select_mate(options)
            if mate is not None:
                # print("got one!")
                offspring = self.reproduce(mate)
            else:
                # print("but got none")
                offspring = None
                self.move()  # find more fertile area
        self.age += 1
        # print("returning offspring {}".format(offspring))
        return offspring

    def check_dna_will_reproduce(self):
        return random.random() < self.reproduction_probability

    def is_baby(self):
        return self.age < Person.MINIMUM_REPRODUCTION_AGE

    def maybe_die(self):
        # use self.age as well as a certain gene
        # make sure there are random elements allowing improbable survivals
        improbable_survival = random.random() < 0.1  # x% chance of living another year against the odds
        if improbable_survival:
            return False
        return self.is_old() or self.is_overcrowded()

    def is_old(self):
        # if self.is_baby():
        #     return False
        return random.uniform(0, self.age) > self.genetic_age

    def is_overcrowded(self):
        neighbors = self.land.get_mating_options(self)
        return random.uniform(0, len(neighbors)) > self.max_neighbors

    def maybe_move(self):
        # prevent clones from just clustering together in the same place
        if random.random() < self.wanderlust_probability:
            self.move()

    def move(self):
        if self.is_baby():
            return
        self.location = self.land.get_random_location()

    @staticmethod
    def genetic_distance(dna1, dna2, weights=None):
        assert len(dna1) == len(dna2)
        return sum((dna1[i] != dna2[i]) * weights[i] for i in range(len(dna1))) / len(dna1)

    @staticmethod
    def sum_gene(gene):
        return sum(x == "1" for x in gene)

    @staticmethod
    def value_gene_01(gene):
        return Person.sum_gene(gene) / len(gene)

    @staticmethod
    def value_gene_binary(gene):
        n = 0
        for i in range(len(gene)):
            power = 2 ** i
            digit = int(gene[i])  # little-endian because whatevs
            n += digit * power
        return n

    @staticmethod
    def generate_dna():
        return "".join([random.choice("01") for _ in range(Person.N_BASES)])

    @staticmethod
    def combine_dna(p1, p2):
        assert len(p1) == len(p2)
        s = ""
        for i in range(len(p1)):
            s += random.choice([p1[i], p2[i]])
        return s


class Land:
    def __init__(self, x_max, y_max):
        self.x_max = x_max
        self.y_max = y_max
        self.people = []
        self.year = 0
        self.population_history = [0]
        self.next_person_id = 0

    def populate(self, n_people):
        for _ in range(n_people):
            self.add_random_person()
        self.population_history[-1] = len(self.people)

    def add_random_person(self):
        dna = Person.generate_dna()
        land = self
        location = self.get_random_location()
        p = Person(dna, land, location)
        self.add_person(p)

    def get_random_location(self):
        x = random.uniform(0, self.x_max)
        y = random.uniform(0, self.y_max)
        return (x, y)

    def add_person(self, person):
        self.people.append(person)
        # print("born {} --> population {}".format(person, len(self.people)))

    def remove_person(self, person):
        self.people.remove(person)
        # print("died {} --> population {}".format(person, len(self.people)))

    def get_mating_options(self, person):
        return [p for p in self.people if distance(person.location, p.location) <= person.neighbor_radius]

    def go_to_next_generation(self):
        print("year {}".format(self.year))
        self.move_people()
        self.add_people()
        self.kill_people()
        self.year += 1
        print("made it to year {}. there are now {} people".format(self.year, len(self.people)))
        self.population_history.append(len(self.people))

    def move_people(self):
        for p in self.people:
            p.maybe_move()

    def add_people(self):
        people_to_add = []
        for person in self.people:
            p = person.maybe_reproduce()
            if p is not None:
                # print("got baby {}".format(p))
                people_to_add.append(p)
        for p in people_to_add:
            self.add_person(p)

    def kill_people(self):
        people_to_kill = []
        for person in self.people:
            if person.maybe_die():
                people_to_kill.append(person)
        for person in people_to_kill:
            self.remove_person(person)

    def plot_people_locations(self):
        xs = [p.location[0] for p in self.people]
        ys = [p.location[1] for p in self.people]
        ages = [p.age for p in self.people]
        max_age = max(ages)
        age_cmap = plt.get_cmap("gist_rainbow")
        age_nums = [256 * age / max_age for age in ages]
        colors = [age_cmap(n) for n in age_nums]
        plt.scatter(xs, ys, alpha=0.5, c=colors)

        # circles for their radii
        for p in self.people:
            circle = plt.Circle(p.location, p.neighbor_radius, color="r", alpha=0.05)
            plt.gca().add_artist(circle)

        plt.title("people locations")
        plt.show()

    def plot_ages(self):
        plt.hist([p.age for p in self.people])
        plt.title("ages")
        plt.show()

    def plot_population_history(self):
        xs = list(range(len(self.population_history)))
        plt.plot(xs, self.population_history)
        plt.yscale("log")
        plt.title("population history")
        plt.show()

    def report_genomes(self):
        print("--- genome report at year {} ---".format(self.year))
        if len(self.people) == 0:
            print("everyone is dead")
        else:
            genomes = sorted(p.dna for p in self.people)
            d = {}
            for genome in genomes:
                if genome not in d:
                    d[genome] = 0
                d[genome] += 1
            for genome in sorted(d):
                if d[genome] > 1:
                    print("{} (total clones: {})".format(genome, d[genome]))
                else:
                    print("{} (unique)".format(genome))
            self.plot_genome_average()
        print("--- end of genome report for year {} ---".format(self.year))

    def plot_genome_average(self):
        g = [[] for _ in range(Person.N_BASES)]
        for p in self.people:
            for i, b in enumerate(p.dna):
                g[i].append(int(b))
        g_avgs = []
        g_stds = []
        g_0_count = []
        g_1_count = []
        for lst in g:
            g_avgs.append(np.mean(lst))
            g_stds.append(np.std(lst))
            g_0_count.append(lst.count(0))
            g_1_count.append(lst.count(1))

        digs = math.ceil(math.log10(len(self.people)))
        for i in range(Person.N_BASES):
            s = "bit "
            s += str(i).rjust(2)
            s += ": (0 * "
            s += str(g_0_count[i]).rjust(digs)
            s += ") "
            n_dashes = 25
            s_dashes = "-" * n_dashes
            avg_position = math.floor(g_avgs[i] * n_dashes)
            s_dashes = s_dashes[:avg_position] + "X" + s_dashes[avg_position+1:]
            s += s_dashes
            s += " (1 * "
            s += str(g_1_count[i]).rjust(digs)
            s += ")"
            print(s)

        g_minus_1_sd = [max(0, g_avgs[i] - g_stds[i]) for i in range(Person.N_BASES)]
        g_plus_1_sd = [min(1, g_avgs[i] + g_stds[i]) for i in range(Person.N_BASES)]
        xs = list(range(Person.N_BASES))
        plt.bar(xs, g_plus_1_sd, color="#59ba57")
        plt.bar(xs, g_avgs, color="#e5cd19")
        plt.bar(xs, g_minus_1_sd, color="white")
        plt.title("genome bit values")
        plt.show()

    def report(self):
        self.plot_people_locations()
        self.plot_ages()
        self.report_genomes()


if __name__ == "__main__":
    land = Land(30, 30)
    land.populate(100)
    for _ in range(100):
        land.go_to_next_generation()
        if land.year % 10 == 0:
            # land.report()
            pass
    for person in land.people:
        print(person)
    land.report()
