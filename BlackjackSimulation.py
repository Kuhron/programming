import numpy as np
import pandas as pd

import Card


class BlackjackCard(Card.Card):
    @staticmethod
    def from_card(card):
        return BlackjackCard(card.value, card.suit)

    def get_blackjack_value(self):
        return max(10, self.number)


class Hand:
    def __init__(self, cards):
        self.cards = cards
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

        return hard_value, soft_value

    def update_values(self):
        self.hard_value, self.soft_value = self.get_values()

    def is_soft(self):
        return self.soft_value is not None

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

    def get_action(hand, dealer_card):
        is_pair, is_soft = hand.is_pair(), hand.is_soft()
        strategy_matrix = BasicStrategy.PAIR_MATRIX if is_pair else SOFT_MATRIX if is_soft else HARD_MATRIX
        value = hand.soft_value if is_soft else hand.hard_value  # for AA, hard value = 2, which is the index in PAIR_MATRIX
        row = strategy_matrix[value]
        col_index = DEALER_CARD_TO_INDEX[dealer_card.get_blackjack_value()]
        return row[col_index]


class Table:
    def __init__(self, doubleable_hard_values, minimum_bet, maximum_bet, blackjack_payoff_ratio, insurance_payoff_ratio):
        self.doubleable_hard_values = doubleable_hard_values
        self.minimum_bet = minimum_bet
        self.maximum_bet = maximum_bet
        self.blackjack_payoff_ratio = blackjack_payoff_ratio
        self.insurance_payoff_ratio = insurance_payoff_ratio


def should_take_insurance():
    return False  # unless incorporating card counting


if __name__ == "__main__":
    table = Table(
        doubleable_hard_values = [10, 11],
        minimum_bet = 5,
        maximum_bet = 200,
        blackjack_payoff_ratio = 3/2,
        insurance_payoff_ratio = 2/1,
    )

    shoe = Card.Shoe(6)
    shoe.shuffle()
