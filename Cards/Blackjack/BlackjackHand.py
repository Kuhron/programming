from Cards.Blackjack.BlackjackCard import BlackjackCard


class BlackjackHand:
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
        return [BlackjackHand([card], self.current_bet / 2) for card in self.cards]
