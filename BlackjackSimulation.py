import numpy as np


class Hand:
    def __init__(self, cards):
        self.cards = cards

    def get_values(self):
        # note that there can only be one soft total (minimum hand of AA yields [2, 12, 22], of which only 2 are valid)
        


    def is_soft(self):
        ?