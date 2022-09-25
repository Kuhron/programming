# create stuff from the genome
# and this stuff will in turn do chemistry / ecology things

# take certain substrings and walk a path based on them
# 1 = go forward; 0 = turn left (starting from standard position)
# then get some numbers based on this path
# min/max x/y visited
# max/avg distance from origin
# these numbers are the chemical properties of the "proteins" made by the DNA substrings


import random
import numpy as np
import matplotlib.pyplot as plt
import functools

import RawVariation as rv


def get_chemicals(dna):
    # don't get all substrings
    # start somewhere random, have bias toward shorter strings (exponential decay of how long you'll keep transcribing)
    expected_n = len(dna)
    should_transcribe = lambda: random.random() > 1 / expected_n
    expected_len = 10
    get_next_base = lambda: random.random() > 1 / expected_len
    while should_transcribe():
        i = random.randrange(len(dna))
        s = []
        while get_next_base():
            s.append(dna[i])
            i += 1
            if i >= len(dna):
                break
        properties_list = get_chemical_properties(tuple(s))
        yield properties_list


def get_all_substrings(dna):
    n = len(dna)
    for i in range(n):
        max_len = n - i
        for l in range(1, max_len + 1):
            yield tuple(dna[i : i + l])


def get_path_from_substring(arr):
    x,y = 0, 0
    res = [(x, y)]
    direction = 0
    for bit in arr:
        if bit == 0:
            direction = (direction + 90) % 360
        else:
            dx = 1 if direction == 0 else -1 if direction == 180 else 0
            dy = 1 if direction == 90 else -1 if direction == 270 else 0
            x += dx
            y += dy
            res.append((x,y))
    # print(f"arr {arr} gave path {res}")
    return res


@functools.lru_cache(maxsize=100000)
def get_chemical_properties(arr):
    path = get_path_from_substring(arr)
    path = np.array(path)
    xs = path[:, 0]
    ys = path[:, 1]
    distances = (lambda x,y: (x**2+y**2)**0.5)(*path.T)
    mass = np.mean(distances)
    size = max(distances)
    density = mass / size if size != 0 else 0
    leftness = min(xs) * -1
    rightness = max(xs)
    horizontality = rightness - leftness
    downness = min(ys) * -1
    upness = max(ys)
    verticality = upness - downness

    properties = []
    properties.append(("leftness", leftness))
    properties.append(("rightness", rightness))
    properties.append(("horizontality", horizontality))
    properties.append(("downness", downness))
    properties.append(("upness", upness))
    properties.append(("verticality", verticality))
    properties.append(("mass", mass))
    properties.append(("size", size))
    properties.append(("density", density))

    return properties


def simulate_chemical_reactions(chemicals):
    # pick two random molecules, so more common ones will bump into each other more often
    a,b = random.sample(chemicals, 2)
    # something about they will be attracted if their horizontalities are large and in opposite directions
    # similar with verticality, it acts like a different fundamental force but has same effect
    # larger mass makes the reaction take more energy
    raise NotImplementedError





if __name__ == "__main__":
    dnas = [rv.get_dna(1000) for i in range(10)]
    for dna in dnas:
        counts = {}
        for properties_list in get_chemicals(dna):
            chemical = tuple(tup[1] for tup in properties_list)
            if chemical not in counts:
                counts[chemical] = 0
            counts[chemical] += 1
        properties_keys = [tup[0] for tup in properties_list]
        print(f"\nDNA string produced chemicals:\n{', '.join(properties_keys)}")
        for chemical, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            chemical_str = "[" + ", ".join(f"{x:.4f}" if type(x) in [float, np.float64] else str(x) for x in chemical) + "]"
            print(f"{count} units of {chemical_str}")

# how should chemicals behave? some numbers can determine whether / how often they will react with each other
# reactions should produce a result that has some new properties based on the reactants
# environment that organisms live in should have some effect on both the organism and the chemicals
# e.g. heat can speed up the decomposition of chemicals, aid metabolism, trigger certain reactions, etc.

# how should chemicals affect organisms? I think having them be the vector for ecological dynamics will be easiest
# basically the organisms interact through exchange of chemicals and then some can kill others or such by causing certain reactions
# so the chemicals that are around an organism should affect whether it can reproduce
# some could be mutagens, changing the rate of mutation in DNA transcription
# maybe some chemicals can exist in the environment but aren't made by biology (no DNA sequence can make that path)
# 
