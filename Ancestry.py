import random

def distance(xy1, xy2):
    dx = xy2[0] - xy1[0]
    dy = xy2[1] - xy1[1]
    return (dx**2 + dy**2) ** 0.5

def average(xy1, xy2):
    x = (xy1[0] + xy2[0]) * 0.5
    y = (xy1[1] + xy2[1]) * 0.5
    return (x, y)


class Person:
    def __init__(self, dna, location):
        self.dna = dna
        self.location = location
        self.age = 0

    def __repr__(self):
        return "Person\ncoords {}\nage {}\nDNA {}\n".format(self.location, self.age, self.dna)

    def select_mate(self, options):
        best_score = 0
        best_option = None
        options = options[:]
        random.shuffle(options)
        for option in options:
            score = self.score(option)
            if score > best_score:
                best_score = score
                best_option = option
        return option

    def score(self, other):
        if other is self:
            return 0
        # select part of the dna for use as "similarity", not all of it
        n = 40
        dna1 = self.dna[:n]
        dna2 = other.dna[:n]
        d = Person.genetic_distance(dna1, dna2)
        # optimal value of 0.5 similarity for the relevant string
        return (d) * (1 - d)

    @staticmethod
    def genetic_distance(dna1, dna2):
        assert len(dna1) == len(dna2)
        return sum(dna1[i] != dna2[i] for i in range(len(dna1))) / len(dna1)

    def reproduce(self, mate):
        new_dna = Person.combine_dna(self.dna, mate.dna)
        new_location = average(self.location, mate.location)
        return Person(new_dna, new_location)

    def maybe_reproduce(self, people):
        # decide whether to reproduce (based on something in DNA)
        will_reproduce = True
        p = None
        if will_reproduce:
            mate = self.select_mate(people)
            p = self.reproduce(mate)
        self.age += 1
        return p

    def maybe_die(self):
        # use self.age as well as a certain gene
        gene = self.dna[13:23]
        n = Person.sum_gene(gene)
        return random.uniform(0, self.age) < n

    @staticmethod
    def sum_gene(gene):
        return sum(x == "1" for x in gene)

    @staticmethod
    def generate_dna():
        return "".join([random.choice("01") for _ in range(50)])

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

    def populate(self):
        n_people = 10
        for _ in range(n_people):
            self.add_random_person()

    def add_random_person(self):
        x = random.uniform(0, self.x_max)
        y = random.uniform(0, self.y_max)
        dna = Person.generate_dna()
        p = Person(dna, (x, y))
        self.add_person(p)

    def add_person(self, person):
        self.people.append(person)
        print("there are now {} people".format(len(self.people)))

    def remove_person(self, person):
        self.people.remove(person)

    def get_mating_options(self, person):
        radius = 1
        return [p for p in self.people if distance(person.location, p.location) <= radius]

    def go_to_next_generation(self):
        print("year {}".format(self.year))
        people_to_add = []
        for person in self.people:
            p = person.maybe_reproduce(self.people)
            if p is not None:
                people_to_add.append(p)
        for p in people_to_add:
            self.add_person(p)
        self.kill_people()
        self.year += 1
        # input("made it to year {}. press enter to continue".format(self.year))

    def kill_people(self):
        for person in self.people:
            if person.maybe_die():
                self.remove_person(person)


if __name__ == "__main__":
    land = Land(20, 20)
    land.populate()
    for _ in range(100):
        land.go_to_next_generation()
    for person in land.people:
        print(person)
    print("there are {} people".format(len(land.people)))
