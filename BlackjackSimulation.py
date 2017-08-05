import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Card


class BlackjackCard(Card.Card):
    def __init__(self, card, is_face_up):
        super().__init__(card.value, card.suit)
        self.is_face_up = is_face_up

    def get_blackjack_value(self):
        return min(10, self.number)


class Hand:
    def __init__(self, cards=None):
        self.cards = cards if cards is not None else []
        self.update_values()

    def get_values(self):
        # note that there can only be one soft total (minimum hand of AA yields [2, 12, 22], of which only 2 are valid)
        has_ace = False
        val = 0
        for card in self.cards:
            val += card.get_blackjack_value()
            if card.value == "A":
                has_ace = True

        hard_value = val
        if val <= 11 and has_ace:
            soft_value = val + 10
        else:
            soft_value = None

        print(self.cards, hard_value, soft_value)

        return hard_value, soft_value

    def update_values(self):
        self.hard_value, self.soft_value = self.get_values()

    def is_soft(self):
        return self.soft_value is not None

    def is_pair(self):
        return len(self.cards) == 2 and self.cards[0].value == self.cards[1].value

    def is_blackjack(self):
        return len(self.cards) == 2 and self.hard_value == 21

    def has_busted(self):
        return self.hard_value > 21

    def add_card(self, card):
        self.cards.append(card)
        self.update_values()


class BasicStrategy:
    DEALER_CARD_TO_INDEX = {k: i for i, k in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10, 1])}
    ACTION_DICT = {"S": "stay", "H": "hit", "D": "double", "P": "split", "U": "surrender"}  # for reference

    HARD_MATRIX = {
        #    23456789TA
        21: "SSSSSSSSSS",
        20: "SSSSSSSSSS",
        19: "SSSSSSSSSS",
        18: "SSSSSSSSSS",
        17: "SSSSSSSSSS",
        16: "SSSSSHHHUU",
        15: "SSSSSHHHHH",
        14: "SSSSSHHHHH",
        13: "SSSSSHHHHH",
        12: "SSSSSHHHHH",
        11: "DDDDDDDDDH",
        10: "DDDDDDDDHH",
         9: "HDDDDHHHHH",
         8: "HHHHHHHHHH",
         7: "HHHHHHHHHH",
         6: "HHHHHHHHHH",
         5: "HHHHHHHHHH",
    }

    SOFT_MATRIX = {
        #    23456789TA
        19: "SSSSSSSSSS",
        18: "SDDDDSSHHH",
        17: "HDDDDHHHHH",
        16: "HHDDDHHHHH",
        15: "HHDDDHHHHH",
        14: "HHHDDHHHHH",
        13: "HHHDDHHHHH",
    }

    PAIR_MATRIX = {
        #    23456789TA
         2: "PPPPPPPPPP",
        20: "SSSSSSSSSS",
        18: "PPPPPSPPSS",
        16: "PPPPPPPPPP",
        14: "PPPPPPHHHH",
        12: "PPPPPHHHHH",
        10: "DDDDDDDDHH",
         8: "HHHPPHHHHH",
         6: "PPPPPPHHHH",
         4: "PPPPPPHHHH",
    }

    @staticmethod
    def get_action(hand, dealer_card):
        is_pair, is_soft = hand.is_pair(), hand.is_soft()
        strategy_matrix = BasicStrategy.PAIR_MATRIX if is_pair else BasicStrategy.SOFT_MATRIX if is_soft else BasicStrategy.HARD_MATRIX
        value = hand.soft_value if is_soft else hand.hard_value  # for AA, hard value = 2, which is the index in PAIR_MATRIX
        row = strategy_matrix[value]
        col_index = BasicStrategy.DEALER_CARD_TO_INDEX[dealer_card.get_blackjack_value()]
        return row[col_index]

    @staticmethod
    def should_take_insurance():
        return False  # unless incorporating card counting


class Table:
    def __init__(self, minimum_bet, maximum_bet, blackjack_payoff_ratio, insurance_payoff_ratio, n_decks,
                 doubleable_hard_values, double_after_split, hit_more_than_once_after_split, cards_face_up, stay_on_soft_17):
        self.minimum_bet = minimum_bet
        self.maximum_bet = maximum_bet
        self.blackjack_payoff_ratio = blackjack_payoff_ratio
        self.insurance_payoff_ratio = insurance_payoff_ratio
        self.n_decks = n_decks
        self.doubleable_hard_values = doubleable_hard_values
        self.double_after_split = double_after_split
        self.hit_more_than_once_after_split = hit_more_than_once_after_split
        self.cards_face_up = cards_face_up
        self.stay_on_soft_17 = stay_on_soft_17

        self.shoe = self.get_new_shoe()
        self.dealer = Dealer(self.stay_on_soft_17)

    def get_new_shoe(self):
        shoe = Card.ShoeOfCards(n_decks=self.n_decks, ratio_dealt=np.random.uniform(0.6, 0.8))
        shoe.shuffle()
        return shoe

    def shuffle_shoe(self):
        self.shoe = self.get_new_shoe()


class Player:
    def __init__(self, bankroll):
        self.bankroll = bankroll
        self.hand = Hand()
        self.current_bet = 0

    def decide(self, dealer_card):
        return BasicStrategy.get_action(self.hand, dealer_card)

    def is_broke(self):
        return self.bankroll <= 0

    def bet(self, amount):
        self.current_bet += amount
        self.bankroll -= amount

    def double_bet(self):
        self.bet(self.current_bet)

    def has_blackjack(self):
        return self.hand.is_blackjack()

    def lose_turn(self):
        self.reset()

    def win_turn(self, gross_payoff):
        self.bankroll += gross_payoff
        self.reset()

    def reset(self):
        self.hand = Hand()
        self.current_bet = 0


class Dealer(Player):
    def __init__(self, stay_on_soft_17):
        self.stay_on_soft_17 = stay_on_soft_17
        self.hand = Hand()

    def decide(self):
        hard_value, soft_value = self.hand.get_values()
        if hard_value >= 17 or soft_value >= 18:
            return "S"
        elif self.stay_on_soft_17 and soft_value == 17:
            return "S"
        else:
            return "H"


def add_card(player, deck, is_face_up):
    card = BlackjackCard(next(deck), is_face_up)
    player.hand.add_card(card)

def play_round(player, table, with_other_players=True):
    if with_other_players:
        n_other_players = np.random.choice([0, 1, 2, 3, 4, 5])
        other_players = [Player(table.minimum_bet * np.random.randint(1, 101)) for i in range(n_other_players)]
    else:
        other_players = []
    all_players = other_players + [player]
    all_players += [None] * (6 - len(all_players))
    np.random.shuffle(all_players)  # mutates arg

    # initial deal
    if table.shoe.is_dealt_out():
        table.shuffle_shoe()

    for i in range(2):
        for pl in all_players:
            if pl is None:
                continue
            is_face_up = i == 0 or table.cards_face_up
            add_card(pl, table.shoe, is_face_up)

        # dealer
        is_face_up = i == 1
        add_card(table.dealer, table.shoe, is_face_up)

    is_face_up = True  # all cards that follow
    dealer_card = table.dealer.hand.cards[1]

    # player turns
    for pl in all_players:
        if pl is None:
            continue

        if pl.has_blackjack():
            pl.win_turn(pl.current_bet * table.blackjack_payoff_ratio)
            continue

        decision= pl.decide(dealer_card)
        if decision == "H":
            add_card(pl, table.shoe, is_face_up)
            if pl.hand.has_busted():
                pl.lose_turn()






if __name__ == "__main__":
    table = Table(
        doubleable_hard_values = [10, 11],
        minimum_bet = 5,
        maximum_bet = 200,
        blackjack_payoff_ratio = 3/2,
        insurance_payoff_ratio = 2/1,
        n_decks = 6,
        double_after_split = False,
        hit_more_than_once_after_split = False,
        cards_face_up = True,
        stay_on_soft_17 = True,
    )

    player = Player(60)

    bankrolls = [player.bankroll]
    while True:
        play_round(player, table, with_other_players=False)
        bankrolls.append(player.bankroll)
        if player.is_broke():
            break

    plt.plot(bankrolls)
    plt.show()