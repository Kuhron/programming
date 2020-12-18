# implementing the game Set
# interesting math problems
# e.g. how many cards can you have without a set


import itertools
import random


def is_set(c1, c2, c3):
    assert len(c1) == len(c2) == len(c3)  # n dimensions
    assert c1 != c2
    assert c1 != c3
    assert c2 != c3
    for i in range(len(c1)):
        x1 = c1[i]
        x2 = c2[i]
        x3 = c3[i]
        assert x1 in [0, 1, 2], x1
        assert x2 in [0, 1, 2], x2
        assert x3 in [0, 1, 2], x3
        all_same = x1 == x2 == x3
        all_different = sorted([x1, x2, x3]) == [0, 1, 2]
        is_set_this_dimension = all_same or all_different
        if not is_set_this_dimension:
            return False
    # going to return True, make sure the pairwise determinism works
    assert get_third_from_pair(c1, c2) == c3
    assert get_third_from_pair(c1, c3) == c2
    assert get_third_from_pair(c2, c3) == c1
    return True


def has_set(cards):
    for c1, c2, c3 in itertools.combinations(cards, 3):
        if is_set(c1, c2, c3):
            return True
    return False


def get_deck(n_dimensions):
    if n_dimensions <= 0:
        raise ValueError("need at least one dimension")
    elif n_dimensions == 1:
        cards = []
        for i in [0, 1, 2]:
            tup = (i,)
            cards.append(tup)
        return cards
    else:
        cards = []
        for sub_card in get_deck(n_dimensions - 1):
            for i in [0, 1, 2]:
                tup = (i,) + sub_card
                cards.append(tup)
        assert len(cards) == 3 ** n_dimensions
        return cards


def get_third_from_pair(c1, c2):
    assert len(c1) == len(c2)
    assert c1 != c2
    tup = ()
    for i in range(len(c1)):
        x1 = c1[i]
        x2 = c2[i]
        if x1 == x2:
            val = x1
        else:
            vals = [0, 1, 2]
            val, = [x for x in vals if x not in (x1, x2)]  # implicit unpack len 1
        tup += (val,)
    return tup


def draw_cards_no_set_stochastic(n_dimensions):
    deck = get_deck(n_dimensions)
    cards = []
    while len(deck) > 0:
        c = random.choice(deck)
        cards.append(c)
        deck.remove(c)
        if len(cards) == 1:
            continue  # can't rule out any forbidden cards yet, go ahead and add a second card
        for c1, c2 in itertools.combinations(cards, 2):
            third = get_third_from_pair(c1, c2)
            assert third not in cards, "oops, there's a bug"
            if third in deck:
                deck.remove(third)
    return cards


def get_max_no_set_stats(n_dimensions):
    counts = {}
    max_count = 0
    max_example = None
    for i in range(1000):
        cards_no_set = draw_cards_no_set_stochastic(n_dimensions)
        count = len(cards_no_set)
        if count > max_count:
            max_count = count
            max_example = sorted(cards_no_set)
        if count not in counts:
            counts[count] = 0
        counts[count] += 1
    return counts, max_example


if __name__ == "__main__":
    # the example I made manually on paper that I think has 17 cards
    cards_17_no_set = [
        (0,0,0,0),
        (0,0,0,2),
        (0,0,1,0),
        (0,0,2,2),
        (0,1,0,0),
        (0,1,1,0),
        (0,2,1,2),
        (0,2,2,2),
        (1,0,2,2),
        (1,1,0,0),
        (1,1,0,1),
        (1,1,1,0),
        (1,2,1,2),
        (2,0,0,0),
        (2,0,0,2),
        (2,1,0,1),
        (2,1,1,2),
    ]
    assert len(cards_17_no_set) == 17
    assert not has_set(cards_17_no_set)  # yep, indeed it lacks a set, so more than 2^d is possible

    for n_dimensions in [1, 2, 3, 4, 5]:
        print("\n-- {} dimensions".format(n_dimensions))
        max_no_set_stats, max_example_cards = get_max_no_set_stats(n_dimensions)
        for k, v in sorted(max_no_set_stats.items(), reverse=True):
            print("{} cards ({} occurrences)".format(k, v))
        print("example of max cards:")
        for c in max_example_cards:
            print(c)
