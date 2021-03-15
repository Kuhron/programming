import random
import string


def get_wordform():
    vowels = "aoeui"
    consonants = [x for x in string.ascii_lowercase if x not in vowels]
    sonorants = "mnlryw"
    c = lambda: random.choice(consonants)
    v = lambda: random.choice(vowels)
    s = lambda: random.choice(sonorants)
    initial = lambda: c() if random.random() < 0.7 else ""
    final = lambda: c() if random.random() < 0.3 else ""
    noninitial_onset = lambda: c()
    nonfinal_coda = lambda: s() if random.random() < 0.2 else ""

    initial_syll = lambda: initial() + v() + nonfinal_coda()
    medial_syll = lambda: noninitial_onset() + v() + nonfinal_coda()
    final_syll = lambda: noninitial_onset() + v() + final()
    sole_syll = lambda: initial() + v() + final()

    w = lambda n: sole_syll() if n <= 1 else initial_syll() + "".join(medial_syll() for n_ in range(n-2)) + final_syll()
    n = random.randint(1, 3)
    return w(n)


