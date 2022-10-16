# Rock Paper Scissors variant made in Taskmaster game with Nicola 2022-10-15
# rules:
# - starting configuration: gator > fish > crawdad > mosquito > human (> gator), and shortest path between two non-adjacent animals, e.g. gator eats fish and crawdad but is eaten by human and mosquito
# - 3 of each card in the deck, so 15 cards
# - 2 players, dealt 7 cards each (so one not in play after the first deal, but can be used in later deals)
# - each turn both players play a card from their hand (revealed after both have chosen their card), whoever wins that hand gets to switch an arrow in the graph of who eats who (each pair is represented by a card saying "X eats Y" on one side and "Y eats X" on the other, with a star denoting the side that should be started on, and so winning player can flip one of these), or can pass if don't want to flip any of the arrows
# - if tie, that animal is removed from play for the rest of the game. Remove the cards telling who it eats and who eats it, and remove any of it from players' hands. The rest of the deal continues until either player is out of cards.
# - if run out of cards, redeal floor(half of cards in play) to each player
# - whoever wins more hands wins

import random
import itertools


ANIMALS = ["G", "F", "C", "M", "H"]
N_PER_ANIMAL = 3


def get_deck():
    return [a for a in ANIMALS for i in range(N_PER_ANIMAL)]


def get_initial_precedences():
    d = {a: {} for a in ANIMALS}
    assert len(ANIMALS) % 2 == 1, "can't have unambiguous shortest path with an even number of animals, since animals will have antipodes"
    k = len(ANIMALS) // 2
    for i in range(len(ANIMALS)):
        for j in range(-k, k+1):
            ai = ANIMALS[i]
            aj = ANIMALS[(i+j) % len(ANIMALS)]
            if j < 0:
                # j is before i, so j eats i (j > i)
                d[ai][aj] = "<"
                d[aj][ai] = ">"
            elif j > 0:
                # i is before j, so i eats j (i > j)
                d[ai][aj] = ">"
                d[aj][ai] = "<"
            else:
                assert ai == aj
                # d[ai][aj] = "="
                # don't put these in the precedence dict actually, just check equality of animals themselves
    return d


def eats(animal_i, animal_j, precedences):
    assert animal_i != animal_j, "don't check for equal animals"
    prec = precedences[animal_i][animal_j]
    return prec == ">"  # if i > j, i is greater than j, i eats j


def get_cards_available_for_opponent(cards_held, cards_played_this_deal, active_animals):
    # which cards could the opponent have?
    possibilities = get_deck()
    for c in cards_held:
        possibilities.remove(c)
    for c in cards_played_this_deal:
        possibilities.remove(c)
    possibilities = [x for x in possibilities if x in active_animals]
    return possibilities


def choose_best_card(cards_held, cards_played_this_deal, precedences, active_animals):
    # do some basic card counting, based on what has already been played and what we know we have
    # just tally up how many things your cards each beat in the precedences, choose randomly among those that beat the most stuff

    cards_available_for_opponent = get_cards_available_for_opponent(cards_held, cards_played_this_deal, active_animals)
    cards_held = list(set(cards_held))  # don't need to evaluate the same animal more than once
    expected_wins_by_card = {c: 0 for c in cards_held}
    for ch in cards_held:
        for co in cards_available_for_opponent:
            if ch == co:
                # currently it doesn't take into account what happens to expected performance if a tie removes the animal from play
                continue
            if eats(ch, co, precedences):
                expected_wins_by_card[ch] += 1
    choices = dict_arg_max(expected_wins_by_card)
    return random.choice(choices)


def get_badness_of_precedence(ai, aj, precedences, cards_held, cards_available_for_opponent):
    assert ai != aj
    my_cards_that_win = 0
    my_cards_that_lose = 0
    their_cards_that_win = 0
    their_cards_that_lose = 0

    for ch in cards_held:
        for co in cards_available_for_opponent:
            should_count = {ch, co} == {ai, aj}  # only care about how this particular pair of animals interacts
            if not should_count:
                continue
            if eats(ch, co, precedences):
                my_cards_that_win += 1
                their_cards_that_lose += 1
            else:
                my_cards_that_lose += 1
                their_cards_that_win += 1
                # I guess we're always double counting by counting both my and their cards but whatever
    return (my_cards_that_lose + their_cards_that_win) - (my_cards_that_win + their_cards_that_lose)


def choose_precedence_to_flip(cards_held, cards_played_this_deal, precedences, active_animals):
    # do what will make your hand best against other cards that are out there
    # or can abstain (return None) if don't want to flip anything

    cards_available_for_opponent = get_cards_available_for_opponent(cards_held, cards_played_this_deal, active_animals)

    # basic idea for now: score precedences on "badness" for you
    # score 1 for each of your cards (counting duplicate animals) that loses because of that precedence, and -1 for each that wins
    # score 1 for each of your opponent's cards that wins because of that precedence, and -1 for each that loses
    # choose among the precedences with highest badness, flip one of them

    badness_by_precedence = {}
    for ai, aj in itertools.combinations(active_animals, 2):
        badness = get_badness_of_precedence(ai, aj, precedences, cards_held, cards_available_for_opponent)
        badness_by_precedence[(ai, aj)] = badness
    max_badness = max(badness_by_precedence.values())

    if max_badness <= 0:
        # all precedences are good for me, don't flip any
        return None
    else:
        choices = dict_arg_max(badness_by_precedence)
        return random.choice(choices)


def simulate_one_game(verbose=True):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    deck = get_deck()
    precedences = get_initial_precedences()

    n_players = 2

    active_animals = [x for x in ANIMALS]
    win_record = []
    hands_won_by_player = [0 for i in range(n_players)]
    deal_i = 0
    while len(active_animals) > 0:
        # the game ends when all animals have been removed

        # deal the cards for this deal
        n_cards_per_player = len(deck) // n_players
        cards_by_player = [[] for i in range(n_players)]
        deck_to_draw_from = [card for card in deck]
        for card_i in range(n_cards_per_player):
            for player_i in range(n_players):
                card = random.choice(deck_to_draw_from)
                cards_by_player[player_i].append(card)
                deck_to_draw_from.remove(card)  # it will only remove one instance, not all equal ones

        # hands within a deal go until someone is out of cards (can happen earlier than expected when animals are removed from play)
        cards_played_this_deal = []
        hand_i = 0
        while all(len(cards) > 0 for cards in cards_by_player):
            vprint(f"\n---- deal {deal_i}, hand {hand_i} ----")
            vprint(f"cards held are now: {cards_by_player}")
            cards_chosen_this_hand_by_player = [None for i in range(n_players)]
            for player_i in range(n_players):
                # choose a card
                cards_held = cards_by_player[player_i]
                card = choose_best_card(cards_held, cards_played_this_deal, precedences, active_animals)
                cards_chosen_this_hand_by_player[player_i] = card
                cards_by_player[player_i].remove(card)
            # now they play the cards and we see who eats who
            if n_players > 2:
                raise ValueError("unsure what to do when comparing cards among >2 players")
                # maybe count who eats the most others or something, but still there could be a tie among non-identical animals this way
            a0 = cards_chosen_this_hand_by_player[0]
            a1 = cards_chosen_this_hand_by_player[1]
            vprint(f"cards played this hand: {cards_chosen_this_hand_by_player}")
            cards_played_this_deal += [a0, a1]
            if a0 == a1:
                # it's a tie, remove this animal from play
                for player_i in range(n_players):
                    while a0 in cards_by_player[player_i]:
                        cards_by_player[player_i].remove(a0)
                while a0 in deck_to_draw_from:
                    deck_to_draw_from.remove(a0)
                while a0 in deck:
                    deck.remove(a0)
                active_animals.remove(a0)
                vprint(f"{a0} was removed from play")
                vprint(f"cards held are now: {cards_by_player}")
            else:
                if eats(a0, a1, precedences):
                    hand_winner = 0
                    winning_animal = a0
                    losing_animal = a1
                else:
                    hand_winner = 1
                    winning_animal = a1
                    losing_animal = a0
                vprint(f"player {hand_winner} won the hand with {winning_animal} {precedences[winning_animal][losing_animal]} {losing_animal}")
                vprint(f"cards held are now: {cards_by_player}")
                hands_won_by_player[hand_winner] += 1
                win_record.append(hand_winner)
                cards_held = cards_by_player[hand_winner]
                precedence_to_flip = choose_precedence_to_flip(cards_held, cards_played_this_deal, precedences, active_animals)
                if precedence_to_flip is not None:
                    ai, aj = precedence_to_flip
                    pij = precedences[ai][aj]
                    pji = precedences[aj][ai]
                    assert set([pij, pji]) == {">", "<"}
                    new_eater = ai if pij == "<" else aj
                    new_eaten = aj if pij == "<" else ai
                    precedences[ai][aj] = ">" if pij == "<" else "<"
                    precedences[aj][ai] = ">" if pji == "<" else "<"
                    vprint(f"precedence {precedence_to_flip} changed to {new_eater} {precedences[new_eater][new_eaten]} {new_eaten}")
                else:
                    vprint("no precedence flipped")
            hand_i += 1

        vprint(f"score at end of this deal: {hands_won_by_player}")
        deal_i += 1

    # announce the winner
    vprint(f"final scores: {hands_won_by_player}")
    vprint(f"win record: {win_record}")
    return win_record


def get_winners(win_record):
    if len(win_record) == 0:
        return []
    ns = set(win_record)
    counts = {n: win_record.count(n) for n in ns}
    return dict_arg_max(counts)


def dict_arg_max(d):
    max_val = max(d.values())
    return sorted([k for k,v in d.items() if v == max_val])


def get_advantage_stats(n_games):
    records = []
    for i in range(n_games):
        if i % 1000 == 0:
            print(f"simulating game {i}/{n_games}")
        win_record = simulate_one_game(verbose=False)
        records.append(win_record)

    # see if there are advantages to things like winning the first hand, first 2 hands, etc.
    conditions = [
        [0], [1],
        [0, 0], [1, 1],
        [0, 1], [1, 0],
        [0, 0, 0], [1, 1, 1],
        [0, 0, 1], [1, 1, 0],
        [0, 1, 0], [1, 0, 1],
        [0, 1, 1], [1, 0, 0],
    ]
    for condition in conditions:
        # records that start in this way
        l = len(condition)
        relevant_records = [rec for rec in records if rec[:l] == condition]
        winners_including_condition = [tuple(get_winners(rec)) for rec in relevant_records]
        # also look at how the rest of the game goes aside from the initial hand wins
        # (since those will skew the win probability toward whoever happened to win first)
        # (we want to know if there is an advantage in the rest of the game due to having won the first hand or two)
        winners_excluding_condition = [tuple(get_winners(rec[l:])) for rec in relevant_records]

        counts_including_condition = {tuple(n): winners_including_condition.count(n) for n in set(winners_including_condition)}
        counts_excluding_condition = {tuple(n): winners_excluding_condition.count(n) for n in set(winners_excluding_condition)}

        counts_including_condition = {0: counts_including_condition.get((0,), 0), 1: counts_including_condition.get((1,), 0), "tie": counts_including_condition.get((), 0) + counts_including_condition.get((0, 1), 0)}
        counts_excluding_condition = {0: counts_excluding_condition.get((0,), 0), 1: counts_excluding_condition.get((1,), 0), "tie": counts_excluding_condition.get((), 0) + counts_excluding_condition.get((0, 1), 0)}
        print(f"condition {condition} has win counts {counts_including_condition} (incl.) and {counts_excluding_condition} (excl.)")



if __name__ == "__main__":
    simulate_one_game()
    get_advantage_stats(n_games=10000)

