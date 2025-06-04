# based on observation that the following place names are, coincidentally, strings of English words:
# - Singapore = sing a pore
# - Germany = germ any
# - Myanmar = my an mar
# - Madagascar = mad a gas car
# likely many others I haven't noticed


import random
import string


class WeightedList:
    def __init__(self, pairs):
        total_weight = 0
        self.items_with_cumulative_weights = []
        for item, weight in pairs:
            if weight <= 0:
                raise ValueError("weights must be strictly positive")
            total_weight += weight
            self.items_with_cumulative_weights.append([item, total_weight])
        self.total_weight = total_weight

    def choose(self):
        r = random.uniform(0, self.total_weight)
        # just linearly search for now
        for item, w in self.items_with_cumulative_weights:
            if w >= r:
                return item
        raise Exception("did not choose anything")


def lowercase_letters_only(s):
    return "".join(x for x in s if x in string.ascii_lowercase)


def get_word_list(max_len):
    with open("cmudict.txt") as f:
        lines = f.readlines()
    raws = [line.split(" ")[0].lower() for line in lines]
    if max_len is not None:
        raws = [s for s in raws if len(s) <= max_len]
    return [lowercase_letters_only(s) for s in raws]


def weight_words(words, power=1):
    # add extra copies of shorter words
    n = max(len(s) for s in words)

    pairs = []
    for word in words:
        weight = (n + 1 - len(word)) ** power
        pairs.append([word, weight])

    wl = WeightedList(pairs)
    return wl


def make_name(weighted_word_list):
    n = random.choice([2,3])
    ws = [weighted_word_list.choose() for i in range(n)]
    s = "".join(ws)
    return s[0].upper() + s[1:]



if __name__ == "__main__":
    max_len = 5
    power = 2.5

    words = get_word_list(max_len)
    weighted_word_list = weight_words(words, power)
    print("press enter after each name to see another one\n")
    while True:
        name = make_name(weighted_word_list)
        input(name + " ")
