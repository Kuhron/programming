import random
import string


class Person:
    def __init__(self):
        self.name = get_random_name()
        self.location = get_random_location()
        self.propensity_to_copy = random.random()
        self.propensity_to_innovate = random.random()
        self.propensity_to_remain_same = random.random()
        self.k_neighbors = random.randint(1, 5)

    def change_name(self, people):
        pc = self.propensity_to_copy
        pi = self.propensity_to_innovate
        pr = self.propensity_to_remain_same
        p_total = pc + pi + pr
        pc1 = pc/p_total
        pi1 = pi/p_total
        pr1 = pr/p_total
        r = random.random()
        if r < pc1:
            self.copy_name(people)
        elif r < pi1:
            self.innovate_name()
        else:
            pass

    def copy_name(self, people):
        neighbors = get_k_nearest_neighbors(self, self.k_neighbors, people)
        chosen = random.choice(neighbors)
        self.name = chosen.name

    def innovate_name(self):
        self.name = get_random_name()



def get_random_name():
    length = random.randint(2, 10)
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


def get_random_location():
    return [random.random() for i in range(2)]


def get_distance_between_people(p1, p2):
    x1, y1 = p1.location
    x2, y2 = p2.location
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5


def get_k_nearest_neighbors(person, k, people):
    distances = [get_distance_between_people(person, p) for p in people]
    tups = sorted(zip(distances, people))
    assert tups[0][0] == 0  # person has zero distance to self
    return [t[1] for t in tups[1:k+1]]  # don't include self


def summarize_names(people):
    print("\n-- name summary")
    d = {}
    for p in people:
        if p.name not in d:
            d[p.name] = 0
        d[p.name] += 1
    tups = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    for t in tups:
        print(t)
    print("-- done summarizing names")


def change_names(people):
    for p in people:
        p.change_name(people)


if __name__ == "__main__":
    people = [Person() for i in range(200)]
    for i in range(100):
        summarize_names(people)
        change_names(people)
