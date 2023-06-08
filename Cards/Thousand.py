# game from PNG

import random
from Card import Card, DeckOfCards, Player, PlayerSet


class Hand:
    # new data structure for this game since the suit doesn't matter and I don't want to keep calling list.count()
    def __init__(self):
        self.counts = {v: 0 for v in Card.VALUES}

    def append(self, c):
        self.counts[c] += 1

    def remove(self, c):
        self.counts[c] -= 1

    def count(self, v):
        return self.counts[v]

    def as_list(self):
        l = []
        for v in Card.VALUES:
            l += [v] * self.counts[v]
        return l

    def __iter__(self):
        return (x for x in self.as_list())

    def __len__(self):
        return sum(self.counts.values())


def get_n_cards_dealt_each(n_players):
    if n_players == 4:
        return 7
    elif n_players == 5:
        return 6
    else:
        raise ValueError("n_players must be 4 or 5")


def hand_repr_sorted(hand):
    # sort by card values, ignore suit
    ss = []
    for v in Card.VALUES:
        n = sum(c == v for c in hand)
        if n > 0:
            ss.append(v * n)
    return " ".join(ss)


def print_hands(players):
    for p in players:
        print(f"{p} has hand {hand_repr_sorted(p.hand)}")


def print_pile(pile):
    print(f"Pile: {hand_repr_sorted(pile)}")


def get_score_of_card(c):
    assert type(c) is str, type(c)
    if c in "23456789":
        return 5
    elif c in "TJQK":
        return 10
    elif c == "A":
        return 20
    else:
        raise ValueError(c)


def get_penalty_of_hand(hand, round_number):
    if round_number == 0:
        return 0
    else:
        return sum(get_score_of_card(c) for c in hand)


def get_score_of_bank(bank):
    return sum(get_score_of_card(c) for c in bank)


def initial_discard(hand):
    # later need to implement some kind of strategic learning in place of this function, this can just be default behavior
    # pick a 5-point card that is alone, or one that you have 3 of
    choices = [v for v in "23456789" if hand.count(v) in [1, 3]]
    if len(choices) == 0:
        # try a low card with 2 count
        choices = [v for v in "23456789" if hand.count(v) == 2]
    if len(choices) == 0:
        # failsafe
        choices = hand.as_list()
    return random.choice(choices)


def get_discard(hand, pile, opponents):
    # if an opponent is close to finishing, you should be more likely to discard high-point cards
    # later need to implement some kind of strategic learning in place of this function, this can just be default behavior
    return random.choice(hand.as_list())



if __name__ == "__main__":
    n_players = 4
    players = [Player(i) for i in range(n_players)]
    for i in range(n_players):
        players[i].hand = Hand()
        players[i].set_opponents([players[j] for j in range(n_players) if j != i])
        players[i].score = 0

    deck = DeckOfCards()
    deck.shuffle()
    n_cards_dealt_each = get_n_cards_dealt_each(n_players)
    for i in range(n_cards_dealt_each):
        for p in players:
            c = next(deck)
            p.put_card_in_hand(c.value)
    print_hands(players)

    pile = Hand()
    round_number = 0
    while True:
        print(f"\nRound {round_number}")
        dealer = round_number % n_players
        starting_player_index = (dealer + 1) % n_players
        for p in players:
            p.bank = Hand()
            discard = initial_discard(p.hand)
            pile.append(discard)
            p.hand.remove(discard)
        print_pile(pile)
        print_hands(players)

        # gameplay of the round
        player_index = starting_player_index
        while True:
            p = players[player_index]
            draw = next(deck).value
            p.hand.append(draw)

            discard = get_discard(p.hand, pile, p.opponents)
            bam = pile.count(discard) == 1  # so after discarding, there will be two of this value and they will "bam"
            pile.append(discard)
            p.hand.remove(discard)
            if len(p.hand) == 0 and bam:
                break

        for p in players:
            gain = get_score_of_bank(p.bank)
            penalty = get_penalty_of_hand(p.hand, round_number)
            p.score += gain - penalty
        round_number += 1
