from Cards.Card import Card


class BlackjackCard(Card):
    def __init__(self, card, is_face_up):
        super().__init__(card.value, card.suit)
        self.is_face_up = is_face_up

    @staticmethod
    def from_str(s):
        card = Card.from_str(s)
        return BlackjackCard(card, True)

    def get_blackjack_value(self):
        return min(10, self.number)

    def __repr__(self):
        # omit suit
        return self.value
