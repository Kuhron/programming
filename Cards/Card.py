import random


class Card:
    VALUES = "23456789TJQKA"
    SUITS = "SHDC"

    def __init__(self, value, suit):
        assert value.upper() in Card.VALUES, "invalid card value"
        self.value = value.upper()
        self.value_index = Card.VALUES.index(self.value)
        assert suit.upper() in Card.SUITS, "invalid suit"
        self.suit = suit.upper()
        self.str = self.value + self.suit
        self.color = "R" if self.suit in "HD" else "B"
        self.majority = "M" if self.suit in "SH" else "m"
        self.number = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1][self.value_index]

    def __repr__(self):
        return self.str

    @staticmethod
    def from_str(s):
        assert len(s) == 2
        return Card(s[0], s[1])  # tempted to write Card(*s)

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.str == other.str
        return NotImplemented


class DeckOfCards:
    def __init__(self):
        self.cards = DeckOfCards.get_all_cards()
        self.generator = self.deal()

    @staticmethod
    def get_all_cards():
        return [Card(value, suit) for value in Card.VALUES for suit in Card.SUITS]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        for card in self.cards:
            yield card

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)


class Player:
    def __init__(self):
        self.hand = []
        self.opponents = []

    def set_opponents(self, opponents):
        self.opponents = opponents

    def take_card(self, opponent):
        card = opponent.choose_card_to_give_away(self)
        opponent.remove_card_from_hand(card)
        self.hand.append(card)

    def choose_card_to_give_away(self, opponent):
        return random.choice(self.hand)

    def put_card_in_hand(self, card):
        self.hand.append(card)

    def remove_card_from_hand(self, card):
        self.hand.remove(card)

    def trade_with_opponent(self, opponent):
        # they should not be able to see the card they get before choosing what to give
        self_to_opponent = self.choose_card_to_give_away(opponent)
        self.remove_card_from_hand(self_to_opponent)
        opponent_to_self = opponent.choose_card_to_give_away(self)
        opponent.remove_card_from_hand(opponent_to_self)
        self.put_card_in_hand(opponent_to_self)
        opponent.put_card_in_hand(self_to_opponent)

    def take_turn(self):
        raise NotImplementedError

    def has_won(self):
        raise NotImplementedError


class PlayerSet:
    def __init__(self, player_type_dict):
        self.setup_players(player_type_dict)
        self.assign_opponents_to_players()
        # self.game_finished = False

    def setup_players(self, player_type_dict):
        self.players = []
        for player_creation_func, n_players in player_type_dict.items():
            self.players += [player_creation_func() for _ in range(n_players)]
        random.shuffle(self.players)

    def assign_opponents_to_players(self):
        for player in self.players:
            player.set_opponents([x for x in self.players if x is not player])

    def play(self):
        while True: # not self.game_finished:
            for player in self.players:
                player.take_turn()
                if player.has_won():
                    # self.game_finished = True
                    break

