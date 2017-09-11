import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Card


class BlackjackCard(Card.Card):
    def __init__(self, card, is_face_up):
        super().__init__(card.value, card.suit)
        self.is_face_up = is_face_up

    @staticmethod
    def from_str(s):
        card = Card.Card.from_str(s)
        return BlackjackCard(card, True)

    def get_blackjack_value(self):
        return min(10, self.number)

    def __repr__(self):
        # omit suit
        return self.value


class Hand:
    def __init__(self, cards=None, bet=0):
        self.cards = cards if cards is not None else []
        self.update_values()
        self.current_bet = bet

    @staticmethod
    def from_str_list(lst):
        cards = []
        for s in lst:
            cards.append(BlackjackCard.from_str(s))
        return Hand(cards)

    @staticmethod
    def card_repr(card):
        return ("" if card.is_face_up else "*") + repr(card)

    def __repr__(self):
        card_reprs = "[" + " ".join(self.card_repr(card) for card in self.cards) + "]"
        return "({} : {} {})".format(card_reprs, self.hard_value, self.soft_value)

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

        return hard_value, soft_value

    def update_values(self):
        self.hard_value, self.soft_value = self.get_values()
        self.max_value = self.hard_value if self.soft_value is None else self.soft_value

        assert len(self.cards) <= 21, "hand has grown out of control!"
        assert self.hard_value <= 30 and (self.soft_value is None or self.soft_value <= 30), "impossible value for hand {}".format(self)

    def is_soft(self):
        return self.soft_value is not None

    def is_pair(self):
        return len(self.cards) == 2 and self.cards[0].value == self.cards[1].value

    def is_blackjack(self):
        return len(self.cards) == 2 and self.max_value == 21

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
    def should_take_insurance(tc):
        # return False  # unless incorporating card counting
        return tc >= 3


class Table:
    def __init__(self, minimum_bet, maximum_bet, blackjack_payoff_ratio, insurance_payoff_ratio, n_decks, max_hands_total,
                 doubleable_hard_values, double_after_split, hit_more_than_once_after_split,
                 cards_face_up, stay_on_soft_17, pay_blackjack_after_split):
        self.minimum_bet = minimum_bet
        self.maximum_bet = maximum_bet
        self.blackjack_payoff_ratio = blackjack_payoff_ratio
        self.insurance_payoff_ratio = insurance_payoff_ratio
        self.n_decks = n_decks
        self.max_hands_total = max_hands_total
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
        vprint("shuffling new shoe")  # debug
        self.shoe = self.get_new_shoe()


class Player:
    def __init__(self, bankroll, is_counting, counting_and_betting_system):
        self.name = None
        self.bankroll = bankroll
        self.hands = [Hand()]
        self.is_counting = is_counting
        self.counting_and_betting_system = counting_and_betting_system
        self.running_count = 0
        self.true_count = 0
        self.insurance_bet = 0

    def __repr__(self):
        return self.name

    def is_dealer(self):
        return False

    def decide(self, hand, dealer_card):
        return BasicStrategy.get_action(hand, dealer_card)

    def is_broke(self):
        return self.bankroll <= 0

    def bet(self, hand, amount):
        amount = min(amount, self.bankroll)
        vprint("{} bet on hand {}; bet {:.2f} -> {:.2f}. bankroll {:.2f} -> {:.2f}".format(
            self, hand, hand.current_bet, hand.current_bet + amount, self.bankroll, self.bankroll - amount))
        hand.current_bet += amount
        self.bankroll -= amount

    def get_initial_bet(self, table):
        ideal_bet = self.get_bet_from_minimum_and_true_count(table.minimum_bet, self.true_count)
        return min(table.maximum_bet, max(table.minimum_bet, ideal_bet))

    def get_bet_from_minimum_and_true_count(self, minimum, tc):
        return self.counting_and_betting_system.get_bet_amount(tc) if self.is_counting else minimum

    def place_initial_bet(self, hand, table):
        self.bet(hand, self.get_initial_bet(table))

    def double_bet(self, hand):
        self.bet(hand, hand.current_bet)

    def halve_bet(self, hand):
        self.bet(hand, -0.5 * hand.current_bet)

    def has_already_split(self):
        return len(self.hands) > 1

    def lose_on_hand(self):
        # forfeit hand.bet
        vprint("{} lost hand. new bankroll {:.0f}".format(self, self.bankroll))
        pass

    def win_on_hand(self, gross_payoff):
        vprint("{} won hand. bankroll {:.0f} -> {:.0f}".format(self, self.bankroll, self.bankroll + gross_payoff))
        self.bankroll += gross_payoff

    def will_take_insurance(self):
        return BasicStrategy.should_take_insurance(self.true_count)

    def has_insurance(self):
        return self.insurance_bet > 0

    def lose_insurance_bet(self):
        self.bankroll -= self.insurance_bet
        vprint("{} loses insurance bet of {:.2f}; bankroll -> {:.2f}".format(self, self.insurance_bet, self.bankroll))
        self.insurance_bet = 0

    def reset(self):
        self.hands = [Hand()]
        self.insurance_bet = 0

    def count(self, card, shoe):
        if self.is_counting:
            self.running_count += self.counting_and_betting_system.get_count_value(card)
            self.true_count = self.get_true_count(shoe)
            vprint("{} counted card {}; rc = {}; {:.2f} decks left => tc = {:.2f}".format(
                self, card, self.running_count, shoe.get_n_decks_left(), self.true_count)
            )

    def get_true_count(self, shoe):
        if self.is_counting:
            decks_left = shoe.get_n_decks_left()
            return self.running_count / decks_left
        return 0

    def reset_count(self):
        self.running_count = 0


class Dealer(Player):
    def __init__(self, stay_on_soft_17):
        super().__init__(np.inf, False, None)
        self.stay_on_soft_17 = stay_on_soft_17
        self.hands = [Hand()]

    def is_dealer(self):
        return True

    def decide(self, hand_arg, dealer_card):
        # ignore hand_arg (since dealer cannot split so only has one hand) and dealer_card
        assert len(self.hands) == 1
        hand = self.hands[0]
        hard_value, soft_value = hand.hard_value, hand.soft_value
        if hard_value >= 17:
            return "S"
        elif soft_value is not None and soft_value >= 18:
            return "S"
        elif self.stay_on_soft_17 and soft_value == 17:
            return "S"
        return "H"

    def has_blackjack(self):
        return self.hands[0].is_blackjack()


class CountingAndBettingSystem:
    def __init__(self, count_function_of_value, bet_function_of_tc):
        self.count_function = count_function_of_value
        self.bet_function_of_tc = bet_function_of_tc

    def get_count_value(self, card):
        return self.count_function(card.get_blackjack_value())

    def get_bet_amount(self, tc):
        return self.bet_function_of_tc(tc)


def add_card(hand, deck, is_face_up, counting_player):
    card = BlackjackCard(next(deck), is_face_up)
    hand.add_card(card)
    vprint("new card in hand {}".format(hand))
    if is_face_up:
        counting_player.count(card, deck)


def play_turn(player, table, dealer_card, counting_player):
    vprint("-- playing turn for {}".format(player.name))
    shoe = table.shoe
    for hand in player.hands:
        while True:
            if hand.has_busted():
                vprint("{} busted with hand {}".format(player, hand))
                break
            elif hand.is_blackjack():
                vprint("{} has blackjack with hand {}".format(player, hand))
                break

            decision = player.decide(hand, dealer_card)

            # restrictions
            if decision == "D" and player.has_already_split() and not table.double_after_split:
                decision = "H"
            if decision == "P" and len(player.hands) >= table.max_hands_total:
                decision = "H"  # TODO is this always true? probably not (e.g. 8s with a high count); treat it as HARD_MATRIX rather than PAIR_MATRIX

            vprint("{} has hand {} and decision {}".format(player, hand, decision))

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
                assert hand in player.hands, "hand {} not in list:\n{}".format(hand, player.hands)
                new_hands = hand.split()
                player.hands.remove(hand)
                for new_hand in new_hands:
                    add_card(new_hand, shoe, is_face_up=True, counting_player=counting_player)
                    player.hands.append(new_hand)
                play_turn(player, table, dealer_card, counting_player)  # replay on the resulting hands
                break  # do not keep looping on old state (before split)
                # TODO add restrictions on play after split, including staying on non-splittable hand and splitting another one after that
                # e.g. dealt AA, split to A A, hit to A9 A, hit next hand to A9 AA, split to A9 A A, hit to A9 AT A, hit to A9 AT A2 (allowed to hit again?)

            else:
                raise Exception("unhandled decision: {}".format(decision))


def reset_all_players(all_players, dealer):
    for pl in all_players:
        pl.reset()
    dealer.reset()


def play_round(player, table, with_other_players=True):
    if with_other_players:
        n_other_players = np.random.choice([0, 1, 2, 3, 4, 5])
        other_players = [Player(table.minimum_bet * np.random.randint(1, 101), False, None) for i in range(n_other_players)]
    else:
        other_players = []
    all_players = other_players + [player]
    np.random.shuffle(all_players)  # mutates arg
    # DO NOT PUT DEALER IN all_players; treat them separately

    for i, pl in enumerate(all_players):
        pl.name = "main player" if pl is player else "other player {}".format(i)

    shoe = table.shoe
    dealer = table.dealer
    dealer.name = "dealer"

    # re-shuffle if necessary
    if shoe.is_dealt_out():
        table.shuffle_shoe()
        shoe = table.shoe  # table.shuffle_shoe() changes table.shoe to a different object, so re-assign this reference
        player.reset_count()
    vprint("{:.2f} decks left in shoe".format(shoe.get_n_decks_left()))

    # initial bet
    for pl in all_players:
        pl.place_initial_bet(pl.hands[0], table)

    # initial deal
    for i in range(2):
        for pl in all_players:
            is_face_up = i == 1 or table.cards_face_up
            add_card(pl.hands[0], shoe, is_face_up, player)

        # dealer
        is_face_up = i == 1
        add_card(dealer.hands[0], shoe, is_face_up, player)

    is_face_up = True  # all cards that follow
    dealer_card = dealer.hands[0].cards[1]
    vprint("dealer shows {}".format(dealer_card))

    if dealer_card.value == "A":
        for pl in all_players:
            # TODO: implement insurance here
            if pl.will_take_insurance():
                assert len(pl.hands) == 1  # no one has had chance to split yet
                pl.insurance_bet = pl.hands[0].current_bet / 2  # half of original bet always, as far as I know
                vprint("{} takes insurance, betting {:.2f}".format(pl, pl.insurance_bet))
                # raise; pl.bet(pl.hands[0], pl.insurance_bet)  # DON'T do this; the player will be overpaid if dealer has blackjack

    if dealer.has_blackjack():
        vprint("dealer has blackjack")
        for pl in all_players:
            assert len(pl.hands) == 1  # no one has had chance to split yet
            hand = pl.hands[0]
            if (not dealer_card.value == "A") and hand.is_blackjack():
                # assume player declared blackjack immediately or cards are face up (some games will only pay even money otherwise)
                pl.win_on_hand(hand.current_bet * (1 + table.blackjack_payoff_ratio))
            # elif dealer_card.value == "A" and pl.has_insurance():  # redundant; can't take insurance unless dealer shows ace
            elif pl.has_insurance():
                pl.win_on_hand(pl.insurance_bet * (table.insurance_payoff_ratio))
                # note do not win back current bet (it is lost since dealer has blackjack)
            else:
                pl.lose_on_hand()
        reset_all_players(all_players, dealer)
        return
    elif dealer_card.value == "A":
        vprint("dealer does not have blackjack")

    for pl in all_players:
        if pl.has_insurance():
            pl.lose_insurance_bet()

    # player turns
    for pl in all_players:
        play_turn(pl, table, dealer_card, player)

    # dealer turn
    # show cards
    for card in dealer.hands[0].cards:
        if not card.is_face_up:
            vprint("dealer flipped over {}".format(card))
            player.count(card, shoe)
            card.is_face_up = True
    play_turn(dealer, table, None, player)

    dealer_hand_value = dealer.hands[0].max_value

    # payoffs
    for pl in all_players:
        for hand in pl.hands:
            if hand.has_busted():  # regardless of dealer outcome
                pl.lose_on_hand()
            elif hand.is_blackjack() and (len(pl.hands) == 1 or table.pay_blackjack_after_split):
                # remember, always show blackjack immediately if you are dealt it! (some tables will only pay even money otherwise)
                pl.win_on_hand(hand.current_bet * (1 + table.blackjack_payoff_ratio))
            elif hand.max_value > dealer_hand_value:
                pl.win_on_hand(hand.current_bet * 2)
            elif hand.max_value == dealer_hand_value:
                # push
                pl.win_on_hand(hand.current_bet)
            else:
                pl.lose_on_hand()

    # count remaining cards
    for pl in all_players:
        for hand in pl.hands:
            for card in hand.cards:
                if not card.is_face_up:
                    card.is_face_up = True  # pointless, but for consistency
                    vprint("{} flipped over {}".format(pl, card))
                    player.count(card, shoe)

    # reset everyone
    reset_all_players(all_players, dealer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", dest="verbose", action="store_true")
    parser.add_argument("-n", dest="n_rounds", type=int, default=100)
    args = parser.parse_args()
    vprint = print if args.verbose else lambda *_, **__: None

    table = Table(
        doubleable_hard_values = [9, 10, 11],  # (orig. [10, 11])
        minimum_bet = 5,  # (orig. 5)
        maximum_bet = 200,  # this may be the biggest factor limiting gains when the count is high (orig. 200)
        blackjack_payoff_ratio = 3/2,  # (orig. 3/2)
        insurance_payoff_ratio = 2/1,  # (orig. 2/1)
        n_decks = 6,
        max_hands_total = 4,  # limit splitting
        double_after_split = True,
        hit_more_than_once_after_split = False,
        cards_face_up = True,
        stay_on_soft_17 = True,
        pay_blackjack_after_split = False,
    )

    def count_function_of_value(n):
        if 2 <= n <= 6:
            return +1
        elif n in [1, 10]:
            return -1
        return 0

    def bet_function_of_tc(tc):
        threshold = 3
        def transform(tc):
            # return 0
            # return np.random.pareto(1 + 1/tc)
            # return tc
            # return tc ** 0.5
            # return np.log(tc)
            # return tc**4
            return np.inf  # will bet table max
        return table.minimum_bet * transform(tc) if tc >= threshold else 0

    counting_and_betting_system = CountingAndBettingSystem(count_function_of_value, bet_function_of_tc)

    player = Player(2 * table.minimum_bet * args.n_rounds, is_counting=True, counting_and_betting_system=counting_and_betting_system)

    bankrolls = [player.bankroll]
    counts = [0]
    true_counts = [0]
    n_rounds = 0
    while True:
        vprint("\n---- round {} ----".format(n_rounds))
        if n_rounds > args.n_rounds:
            break
        play_round(player, table, with_other_players=True)
        bankrolls.append(player.bankroll)
        counts.append(player.running_count)
        true_counts.append(player.true_count)
        if player.is_broke():
            break
        n_rounds += 1

    d_cash = np.diff(np.array(bankrolls))
    ev = np.mean(d_cash)
    sd = np.std(d_cash)
    print("\n\nEV {:.2f} , SD {:.2f}".format(ev, sd))

    # plt.plot(counts, c="r")
    plt.plot(true_counts, c="r")

    ax2 = plt.gca().twinx()
    ax2.plot(bankrolls, c="g")
    plt.show()

    # plt.hist(d_cash, bins=50)
    # plt.show()

    # TODO: be able to reproduce the statistics table at https://wizardofodds.com/games/blackjack/card-counting/high-low/
    # TODO: implement insurance when TC > +3 (players are not yet betting on it even though I put this in the function)