import numpy as np
import random

from Card import Card, DeckOfCards, Player, PlayerSet


# class PlayerSet:
#     def __init__(self, deck_generator, n_manual_players=1, n_ai_players=1):
#         self.players = [ManualPlayer(deck_generator) for i in range(n_manual_players)] + [AiPlayer(deck_generator) for i in range(n_manual_players)]
#         for player in self.players:
#             player.set_opponents([x for x in self.players if x is not player])


class Player:
    def __init__(self, deck_generator, board):
        self.deck_generator = deck_generator
        self.board = board
        self.hand = [next(self.deck_generator) for i in range(5)]
        self.opponents = None

    def set_opponents(self, opponents):
        self.opponents = opponents

    def get_position(self):
        return self.get_position_from_cards(self.hand)

    def get_position_from_cards(self, cards):
        return sum([(-1 if card.color == "R" else 1) * card.number for card in cards])

    def get_color_mean_diff(self):
        b_cards = [card.number for card in self.hand if card.color == "B"]
        r_cards = [card.number for card in self.hand if card.color == "R"]
        if b_cards == [] or r_cards == []:
            return np.nan
        return np.mean(b_cards) - np.mean(r_cards)

    def get_black_ratio(self):
        return sum([1 for card in self.hand if card.color == "B"]) / len(self.hand)

    def get_action(self):
        raise NotImplementedError

    def draw(self, verbose=False):
        card = next(self.deck_generator)
        if verbose:
            print("You drew: {0}".format(card))
        self.hand.append(card)

    def steal(self, verbose=False):
        self.draw(verbose=verbose)
        self.take_card_from_player(random.choice(self.opponents), other_has_choice=False)
        if verbose:
            print("You stole: {0}".format(stolen_card))

    def trade(self, verbose=False):
        card = self.get_card_from_hand()
        counterparty = random.choice(self.opponents)
        self.give_card_to_player(card, counterparty)
        self.take_card_from_player(counterparty, other_has_choice=True)

    def lock(self):
        cards = self.get_subset_adding_to_zero()
        if cards is not None:
            self.put_cards_on_board(cards)
        else:
            print("forfeiting lock turn due to no subset found; write better way to know whether locking is possible before deciding to do it")  # TODO

    def get_card_from_hand(self):
        return random.choice(self.hand)

    def choose_card_to_give(self):
        # TODO improve
        return random.choice(self.hand)

    def get_subset_adding_to_zero(self):
        for i in range(1000):
            n = random.choice([2, 3, 4])
            cards = random.sample(self.hand, n)
            if self.get_position_from_cards(cards) == 0:
                return cards

    def put_cards_on_board(self, cards):
        for c in cards:
            self.board.add_card(c)

    def give_card_to_player(card, counterparty):
        self.hand.remove(card)
        counterparty.hand.append(card)

    def take_card_from_player(counterparty, other_has_choice):
        if other_has_choice:
            chosen = counterparty.choose_card_to_give()
        else:
            chosen = counterparty.get_card_from_hand()
        counterparty.hand.remove(chosen)
        self.hand.append(chosen)


class ManualPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_action(self):
        self.report_status()
        a = input("Your turn.\n1. Draw\n2. Lock in a set adding to zero\n3. Trade\n4. Steal\nselection: ")
        if a == "1":
            self.draw(verbose=True)
        elif a == "2":
            self.lock()
        elif a == "3":
            self.trade(verbose=True)
        elif a == "4":
            self.steal(verbose=True)
        else:
            print("Invalid selection. Try again.\n")
            self.get_action()
        self.report_status()

    def report_status(self):
        print("Hand:", self.hand)
        print("position (p):", self.get_position())
        print("color mean diff (q):", self.get_color_mean_diff())
        print("black ratio (r):", self.get_black_ratio())
        print("")

    def get_card_from_hand(self):
        a = input("Card to get rid of: ")
        cards = [x for x in self.hand if repr(x) == a]
        if len(cards) == 0:
            print("You don't have that card.")
            return self.get_card_from_hand()
        elif len(cards) > 1:
            raise RuntimeError("multiple identical cards found in hand; please debug")
        return cards[0]


class AiPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_action(self):
        a = random.choice(["draw", "lock", "trade", "steal"])
        print("AiPlayer's action:", a)
        if a == "draw":
            self.draw()
        elif a == "lock":
            self.lock()
        elif a == "trade":
            self.trade()
        elif a == "steal":
            self.steal()


class Board:
    def __init__(self):
        self.cards = {v: [] for v in Card.VALUES}

    def add_card(self, card):
        self.cards[card.value].append(card.suit)

    def __repr__(self):
        s = ""
        s += " ".join(["."] + [i for i in Card.VALUES]) + "\n"
        for suit in Card.SUITS:
            s += " ".join([suit] + ["X" if suit in self.cards[v] else " " for v in Card.VALUES]) + "\n"
        return s


def play(player_set, board):
    while True:
        for player in player_set.players:
            player.get_action()
            print(board)
            if player.get_position() == 0:
                print("player won:", player)
                return


if __name__ == "__main__":
    deck = DeckOfCards()
    deck.shuffle()
    deck_generator = deck.deal()
    board = Board()
    n_manual_players = 1
    n_ai_players = 1
    create_manual_player = lambda: ManualPlayer(deck_generator, board)
    create_ai_player = lambda: AiPlayer(deck_generator, board)
    player_type_dict = {
        create_manual_player: n_manual_players,
        create_ai_player: n_ai_players,
    }
    player_set = PlayerSet(player_type_dict)
    play(player_set, board)
