import random
import string
import numpy as np
import matplotlib.pyplot as plt


class Person:
    def __init__(self):
        self.name = get_random_name()
        self.location = get_random_location()
        self.propensity_to_copy = 10  # random.random()
        self.propensity_to_innovate = 1  # random.random()
        self.propensity_to_remain_same = 3  # random.random()
        self.k_neighbors = random.randint(3, 10)

    def change_name(self, people):
        pc = self.propensity_to_copy
        pi = self.propensity_to_innovate
        pr = self.propensity_to_remain_same
        p_total = pc + pi + pr
        pc1 = pc/p_total
        pi1 = pi/p_total
        pr1 = pr/p_total

        fs = [lambda: self.copy_name(people), lambda: self.innovate_name(), lambda: None]
        choice = np.random.choice(fs, p=[pc1, pi1, pr1])
        choice()

    def copy_name(self, people):
        neighbors = get_k_nearest_neighbors(self, self.k_neighbors, people)
        chosen = random.choice(neighbors)
        self.name = chosen.name

    def innovate_name(self):
        self.name = get_random_name()
        # print("innovated {}".format(self.name))



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


def get_count_dict(lst):
    d = {}
    for x in lst:
        if x not in d:
            d[x] = 0
        d[x] += 1
    return d


def summarize_names(people, top_n=None):
    print("\n-- name summary")
    names = [p.name for p in people]
    d = get_count_dict(names)
    tups = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        tups = tups[:top_n]
    for t in tups:
        print(t)
    print("-- done summarizing names")


def change_names(people):
    for p in people:
        p.change_name(people)


def plot_counts(count_dicts):
    all_names = set()
    for d in count_dicts:
        all_names |= set(d.keys())
    for name in all_names:
        counts = [d.get(name, 0) for d in count_dicts]
        plt.plot(counts)
    plt.show()


if __name__ == "__main__":
    people = [Person() for i in range(200)]
    count_dicts = []
    for i in range(200):
        count_dict = get_count_dict([p.name for p in people])
        count_dicts.append(count_dict)
        summarize_names(people, top_n=10)
        change_names(people)
        # input("press enter to continue")

    plot_counts(count_dicts)
