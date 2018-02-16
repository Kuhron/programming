from Cards.Card import DeckOfCards
from Cards.Blackjack.CountingAndBettingSystem import CountingAndBettingSystem
from Cards.Blackjack.BlackjackCard import BlackjackCard


class Shoe(DeckOfCards):
    def __init__(self, n_decks, ratio_dealt):
        super().__init__()
        self.n_decks = n_decks
        self.cards = [BlackjackCard(x, False) for x in DeckOfCards.get_all_cards() * self.n_decks]
        self.cards_dealt = []
        self.cards_left = []
        self.n_cards = len(self.cards)
        self.ratio_dealt = ratio_dealt
        self.n_cards_dealt = 0

    def deal(self):
        self.shuffle()
        self.cards_left = [x for x in self.cards]
        for card in self.cards:
            self.cards_dealt.append(card)
            self.cards_left = self.cards_left[1:]
            self.n_cards_dealt += 1
            yield card

    def is_dealt_out(self):
        return self.n_cards_dealt >= self.ratio_dealt * self.n_cards

    def get_n_decks_left(self):
        decks_dealt = self.n_cards_dealt / 52
        decks_left = self.n_decks - decks_dealt
        return decks_left

    def get_n_cards_left(self):
        return 52 * self.n_decks - self.n_cards_dealt

    def get_hi_lo_count(self):
        running_count = sum(CountingAndBettingSystem.get_hi_lo_count_value(x) for x in self.cards_dealt)
        return running_count / self.get_n_decks_left()



