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
