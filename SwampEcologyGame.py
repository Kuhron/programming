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


ANIMALS = ["G", "F", "C", "M", "H"]


def get_deck():
    return [a for a in ANIMALS for i in range(3)]


def get_initial_precedences():
    d = {a: {} for a in ANIMALS}
    for i in range(len(ANIMALS)):
        for j in range(-2, 2+1):
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


def choose_best_card(cards_held, cards_played_this_deal, precedences, active_animals):
    # can add some card counting later, based on what has already been played and what we know we have
    # just tally up how many things your cards each beat in the precedences, choose randomly among those that beat the most stuff

    # first pass
    return random.choice(cards_held)


def choose_precedence_to_flip(cards_held, cards_played_this_deal, precedences, active_animals):
    # do what will make your hand best against other cards that are out there
    # or can abstain (return None) if don't want to flip anything

    # first pass
    return random.sample(active_animals, 2) if random.random() < 0.8 else None



if __name__ == "__main__":
    deck = get_deck()
    precedences = get_initial_precedences()

    n_players = 2

    active_animals = [x for x in ANIMALS]
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
            print(f"\n---- deal {deal_i}, hand {hand_i} ----")
            print(f"cards held are now: {cards_by_player}")
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
            print(f"cards played this hand: {cards_chosen_this_hand_by_player}")
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
                print(f"{a0} was removed from play")
            else:
                if eats(a0, a1, precedences):
                    hand_winner = 0
                    winning_animal = a0
                    losing_animal = a1
                else:
                    hand_winner = 1
                    winning_animal = a1
                    losing_animal = a0
                print(f"player {hand_winner} won the hand with {winning_animal} vs {losing_animal}")
                hands_won_by_player[hand_winner] += 1
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
                    print(f"precedence {precedence_to_flip} changed to {new_eater} {precedences[new_eater][new_eaten]} {new_eaten}")
                else:
                    print("no precedence flipped")
            hand_i += 1

        print(f"score at end of this deal: {hands_won_by_player}")
        deal_i += 1

    # announce the winner
    print(f"final scores: {hands_won_by_player}")

