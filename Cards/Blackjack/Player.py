import logging

from Cards.Blackjack.BlackjackHand import BlackjackHand as Hand
from Cards.Blackjack.Strategy import BasicStrategy

logger = logging.getLogger(__name__)


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
        logger.info("{} bet on hand {}; bet {:.2f} -> {:.2f}. bankroll {:.2f} -> {:.2f}".format(
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
        logger.info("{} lost hand. new bankroll {:.2f}".format(self, self.bankroll))
        pass

    def win_on_hand(self, gross_payoff):
        logger.info("{} won hand. payoff {:.2f}. bankroll {:.2f} -> {:.2f}".format(self, gross_payoff, self.bankroll, self.bankroll + gross_payoff))
        self.bankroll += gross_payoff

    def will_take_insurance(self):
        return BasicStrategy.should_take_insurance(self.true_count)

    def has_insurance(self):
        return self.insurance_bet > 0

    def lose_insurance_bet(self):
        self.bankroll -= self.insurance_bet
        logger.info("{} loses insurance bet of {:.2f}; bankroll -> {:.2f}".format(self, self.insurance_bet, self.bankroll))
        self.insurance_bet = 0

    def reset(self):
        self.hands = [Hand()]
        self.insurance_bet = 0

    def count(self, card, shoe):
        if self.is_counting:
            count_change = self.counting_and_betting_system.get_count_value(card)
            count_change_str = "+1" if count_change == 1 else str(count_change)
            self.running_count += count_change
            self.true_count = self.get_true_count(shoe)
            logger.info("{} counted card {} ({}); rc = {}; {:.2f} decks left => tc = {:.2f}".format(
                self, card, count_change_str, self.running_count, shoe.get_n_decks_left(), self.true_count)
            )

    def get_true_count(self, shoe):
        if self.is_counting:
            decks_left = shoe.get_n_decks_left()
            return self.running_count / decks_left
        return 0

    def reset_count(self):
        self.running_count = 0
