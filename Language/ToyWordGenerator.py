import random


def get_syllable():
    onsets = "ptkmnshywlr "
    vowels = "aeiou"
    # codas = "mnslryw "
    codas = onsets
    o = random.choice(onsets)
    v = random.choice(vowels)
    c = random.choice(codas)
    return (o + v + c).replace(" ","")


def get_morpheme(n_syllables=None, exclusion_list=None):
    if n_syllables is None:
        n_syllables = random.choice([1]*4 + [2]*2 + [3] * 1)
    while True:
        res = "".join(get_syllable() for i in range(n_syllables))
        if exclusion_list is None or res not in exclusion_list:
            return res


def get_unique_morphemes(n_morphemes):
    res = []
    while len(res) < n_morphemes:
        m = get_morpheme()
        if m not in res:
            res.append(m)
    return res

