class CountingAndBettingSystem:
    def __init__(self, count_function_of_value, bet_function_of_tc):
        self.count_function = count_function_of_value
        self.bet_function_of_tc = bet_function_of_tc

    def get_count_value(self, card):
        return self.count_function(card.get_blackjack_value())

    def get_bet_amount(self, tc):
        return self.bet_function_of_tc(tc)

    @staticmethod
    def hi_lo_count_function_of_value(n):
        if 2 <= n <= 6:
            return +1
        elif n in [1, 10]:
            return -1
        return 0

    @staticmethod
    def get_hi_lo_count_value(card):
        return CountingAndBettingSystem.hi_lo_count_function_of_value(card.get_blackjack_value())
