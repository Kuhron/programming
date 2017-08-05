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
    def __init__(self, cards=None, bet=0):
        self.cards = cards if cards is not None else []
        self.update_values()
        self.current_bet = bet

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

        # print(self.cards, hard_value, soft_value)
        if len(self.cards) > 21:
            raise Exception("hand has grown out of control!")

        return hard_value, soft_value

    def update_values(self):
        self.hard_value, self.soft_value = self.get_values()
        self.max_value = self.hard_value if self.soft_value is None else self.soft_value

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

    def add_bet(self, amount):
        self.current_bet += amount

    def split(self):
        assert len(self.cards) == 2
        assert self.current_bet is not None and self.current_bet > 0
        return [Hand([card], self.current_bet / 2) for card in self.cards]


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
        21: "SSSSSSSSSS",
        20: "SSSSSSSSSS",
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
                 doubleable_hard_values, double_after_split, hit_more_than_once_after_split, cards_face_up, stay_on_soft_17, pay_blackjack_after_split):
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
        self.pay_blackjack_after_split = pay_blackjack_after_split

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
        self.name = str(np.random.randint(0, 10**9))
        self.bankroll = bankroll
        self.hands = [Hand()]
        self.running_count = 0

    def __repr__(self):
        return self.name

    def is_dealer(self):
        return False

    def decide(self, hand, dealer_card):
        return BasicStrategy.get_action(hand, dealer_card)

    def is_broke(self):
        return self.bankroll <= 0

    def bet(self, hand, amount):
        hand.current_bet += amount
        self.bankroll -= amount

    def get_initial_bet(self, table):
        # return table.minimum_bet
        return min(table.maximum_bet, table.minimum_bet * min(1, 1 + self.get_true_count(table.shoe)))

    def place_initial_bet(self, hand, table):
        self.bet(hand, self.get_initial_bet(table))

    def double_bet(self, hand):
        self.bet(hand, hand.current_bet)

    def halve_bet(self, hand):
        self.bet(hand, -0.5 * hand.current_bet)

    def lose_on_hand(self):
        # forfeit hand.bet
        pass

    def win_on_hand(self, gross_payoff):
        self.bankroll += gross_payoff

    def reset(self):
        self.hands = [Hand()]

    def get_count_value(self, card):
        n = card.get_blackjack_value()
        if n <= 6:
            return +1
        elif n in [1, 10]:
            return -1
        return 0

    def count(self, card):
        self.running_count += self.get_count_value(card)

    def get_true_count(self, shoe):
        decks_dealt = shoe.n_cards_dealt / 52
        decks_left = shoe.n_decks - decks_dealt
        return self.running_count / decks_left


class Dealer(Player):
    def __init__(self, stay_on_soft_17):
        super().__init__(np.inf)
        self.stay_on_soft_17 = stay_on_soft_17
        self.hand = Hand()

    def is_dealer(self):
        return True

    def decide(self, hand, dealer_card):
        # ignore hand (since dealer cannot split so only has one hand) and dealer_card
        hard_value, soft_value = self.hand.hard_value, self.hand.soft_value
        if hard_value >= 17 or (soft_value is not None and soft_value >= 18):
            return "S"
        elif self.stay_on_soft_17 and soft_value == 17:
            return "S"
        return "H"

    def has_blackjack(self):
        return self.hands[0].is_blackjack()


def add_card(hand, deck, is_face_up, counting_player):
    card = BlackjackCard(next(deck), is_face_up)
    hand.add_card(card)
    if is_face_up:
        counting_player.count(card)


def play_turn(player, shoe, dealer_card, counting_player):
    for hand in player.hands:
        while True:
            if hand.has_busted() or hand.is_blackjack():
                break
            decision = player.decide(hand, dealer_card)
            print("{} {} has hand {} and decision {}".format(("dealer" if player.is_dealer() else "player"), player, hand.cards, decision))
            if decision == "H":
                add_card(hand, shoe, is_face_up=True, counting_player=counting_player)
            elif decision == "S":
                break
            elif decision == "D":
                player.double_bet(hand)
                # one card after double
                add_card(hand, shoe, is_face_up=True, counting_player=counting_player)
                break
            elif decision == "U":
                # hack up surrender as reducing bet to half and replacing hand with a busted one, so this bet will be lost
                # hopefully I don't implement counting in such a way as to screw it up here (player counting the bogus hand)
                QUEEN_OF_SPADES = BlackjackCard(Card.Card("Q", "S"), is_face_up=NotImplemented)
                player.halve_bet(hand)
                player.hands.remove(hand)
                player.hands.append(Hand([QUEEN_OF_SPADES] * 3))
                break
            elif decision == "P":
                new_hands = hand.split()
                player.hands.remove(hand)
                for new_hand in new_hands:
                    add_card(new_hand, shoe, is_face_up=True, counting_player=counting_player)
                play_turn(player, shoe, dealer_card, counting_player)  # replay on the resulting hands
                # TODO add restrictions on play after split, including staying on non-splittable hand and splitting another one after that
                # e.g. dealt AA, split to A A, hit to A9 A, hit next hand to A9 AA, split to A9 A A, hit to A9 AT A, hit to A9 AT A2 (allowed to hit again?)
                # TODO add option for whether blackjack pays 3:2 after split (often it is treated as just regular 21, so no)
            else:
                raise Exception("unhandled decision: {}".format(decision))


def play_round(player, table, with_other_players=True):
    if with_other_players:
        n_other_players = np.random.choice([0, 1, 2, 3, 4, 5])
        other_players = [Player(table.minimum_bet * np.random.randint(1, 101)) for i in range(n_other_players)]
    else:
        other_players = []
    all_players = other_players + [player]
    np.random.shuffle(all_players)  # mutates arg
    # DO NOT PUT DEALER IN all_players; treat them separately

    shoe = table.shoe
    dealer = table.dealer

    # initial bet
    for pl in all_players:
        pl.place_initial_bet(pl.hands[0], table)

    # initial deal
    if shoe.is_dealt_out():
        table.shuffle_shoe()

    for i in range(2):
        for pl in all_players:
            is_face_up = i == 0 or table.cards_face_up
            add_card(pl.hands[0], shoe, is_face_up, player)

        # dealer
        is_face_up = i == 1
        add_card(dealer.hands[0], shoe, is_face_up, player)

    is_face_up = True  # all cards that follow
    dealer_card = dealer.hands[0].cards[1]

    # player turns
    for pl in all_players:
        play_turn(pl, shoe, dealer_card, player)

    # dealer turn
    # show cards
    for card in dealer.hand.cards:
        card.is_face_up = True
        player.count(card)
    play_turn(dealer, shoe, None, player)

    if dealer.has_blackjack():
        for pl in players:
            pl.lose_turn()
        return

    dealer_hand_value = dealer.hand.max_value

    # payoffs
    for pl in all_players:
        for hand in pl.hands:
            if hand.has_busted():  # regardless of dealer outcome
                pl.lose_on_hand()
            elif hand.is_blackjack() and (len(pl.hands) == 1 or table.pay_blackjack_after_split):
                # remember, always show blackjack immediately if you are dealt it! (some tables will only pay even money otherwise)
                pl.win_on_hand(hand.current_bet * table.blackjack_payoff_ratio)
            elif hand.max_value > dealer_hand_value:
                pl.win_on_hand(hand.current_bet * 2)

    # count remaining cards
    for pl in all_players:
        for hand in pl.hands:
            for card in hand.cards:
                if not card.is_face_up:
                    card.is_face_up = True  # pointless, but for consistency
                    player.count(card)

    # reset everyone
    for pl in all_players:
        pl.reset()
    dealer.reset()


if __name__ == "__main__":
    table = Table(
        doubleable_hard_values = [10, 11],
        minimum_bet = 5,
        maximum_bet = 200,
        blackjack_payoff_ratio = 1 + 3/2,
        insurance_payoff_ratio = 2/1,
        n_decks = 6,
        double_after_split = False,
        hit_more_than_once_after_split = False,
        cards_face_up = True,
        stay_on_soft_17 = True,
        pay_blackjack_after_split = False,
    )

    player = Player(60)

    bankrolls = [player.bankroll]
    n_rounds = 0
    while True:
        if n_rounds > 1000:
            break
        play_round(player, table, with_other_players=False)
        bankrolls.append(player.bankroll)
        if player.is_broke():
            break
        n_rounds += 1

    plt.plot(bankrolls)
    plt.show()