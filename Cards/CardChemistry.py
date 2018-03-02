# second attempt (first = ParticleArray.py in main programming directory)

# want molecule size limited (at least have lower probability the larger the molecule gets; most rulesets so far create infinite growth after critical mass/structure is reached)


import random
import numpy as np

import Cards.Card as Card


class CardParticleArray:
    def __init__(self, side_length):
        self.side_length = side_length
        self.array = np.array([[None] * side_length] * side_length)
        self.started = False
        self.finished = False

    def add_card(self, card):
        # place it in the spot with best score
        best_score = 0
        best_spots = []
        for r in range(self.side_length):
            for c in range(self.side_length):
                score = self.get_score(card, r, c)
                if score > best_score:
                    best_score = score
                    best_spots = [(r, c)]
                elif score == best_score:
                    best_spots.append((r, c))
                else:
                    pass

        # if nothing is sufficient, do not place card
        if best_score == 0 and self.started:
            self.finished = True
            return

        if not self.started:
            self.started = True
            half = int(self.side_length / 2)
            r = c = half
        else:
            # assert len(best_spots) == 1  # best spot should be unique, even if resorting to card value to disambiguate, but ideally only using suits
            # r, c = best_spots[0]
            r, c = random.choice(best_spots)

        self.array[r, c] = card

    def get_score(self, card, row, col):
        # score of placing the card at (row, col) in the array
        # higher is better
        if self.card_at(row, col) is not None:
            return 0

        neighbor_coords = self.get_neighbors(row, col)
        MAX_SATURATION = 3
        MAX_SATISFACTION = 3

        if any(x >= MAX_SATURATION for x in [self.get_saturation(r, c) for r, c in neighbor_coords]):
            # if any neighbor is already saturated, cannot place the card
            return 0
        if any(x >= MAX_SATISFACTION for x in [self.get_satisfaction(r, c) for r, c in neighbor_coords]):
            # if any neighbor is already satisfied
            return 0
        if any(x > MAX_SATISFACTION for x in [self.get_satisfaction(r, c) + self.neighbor_value(card, self.card_at(r, c)) for r, c in neighbor_coords]):
            # if placement of card will over-satisfy a neighbor
            return 0
 
        neighbor_effect = sum(self.neighbor_value(card, self.card_at(r, c)) for r, c in neighbor_coords)
        return neighbor_effect
        

    def get_neighbors(self, row, col):
        return [
            (row + 1, col), (row - 1, col),
            (row, col + 1), (row, col - 1),
        ]

    def get_saturation(self, row, col):
        if self.card_at(row, col) is None:
            return 0
        neighbor_coords = self.get_neighbors(row, col)
        result = 0
        for r, c in neighbor_coords:
            if self.card_at(r, c) is not None:
                result += 1
        return result

    def get_satisfaction(self, row, col):
        if self.card_at(row, col) is None:
            return 0
        neighbor_coords = self.get_neighbors(row, col)
        result = 0
        for r, c in neighbor_coords:
            if self.card_at(r, c) is not None:
                result += self.neighbor_value(self.card_at(row, col), self.card_at(r, c))
        return result

    def neighbor_value(self, card1, card2):
        # should be symmetric function
        if card1 is None or card2 is None:
            return 0
        c1 = card1.color
        c2 = card2.color
        m1 = card1.majority
        m2 = card2.majority
        return int(c1 != c2) + int(m1 != m2)

    def card_at(self, row, col):
        return self.array[row % self.side_length, col % self.side_length]

    def print(self):
        s = ""
        s += "-" * (self.side_length * 3) + "\n"
        for r in range(self.side_length):
            for c in range(self.side_length):
                card = self.card_at(r, c)
                if card is None:
                    s += "  "
                else:
                    s += repr(card)
                s += " "
            s += "\n"
        s += "-" * (self.side_length * 3) + "\n"
        print(s)


def get_new_card():
    return np.random.choice(Card.DeckOfCards.get_all_cards())


if __name__ == "__main__":
    array = CardParticleArray(35)
    i = 0
    while not array.finished:
        card = get_new_card()
        array.add_card(card)
        array.print()
        i += 1
        print("iterations: " + str(i))
