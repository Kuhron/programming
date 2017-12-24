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
        shoe = Card.ShoeOfCards(n_decks=self.n_decks, ratio_dealt=np.random.uniform(0.75, 0.9))
        shoe.shuffle()
        return shoe

    def shuffle_shoe(self):
        vprint("shuffling new shoe")  # debug
        self.shoe = self.get_new_shoe()
