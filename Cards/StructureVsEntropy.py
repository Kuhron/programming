rule_str = """
Object of game is to build as much "structure" as you can
before entropy wins out. Structures are of various types,
such as 2 5 8 J, all diamonds, 7 of each suit, etc.
(need these to be more clearly defined for scoring)
There are 12 card values that use modular arithmetic.
K is 0/12, A is 1, and J is 11.
Queens are all agents of entropy,
and the Queen of Spades is Eris herself.
Once Eris appears, the game ends and no further structure
can be built.
Score is based on structure points - entropy created.
Structure should be defined based on probability of the hand.
Entropy is number of cards seen.
"""

import numpy as np

import Card

# need to check if hand matches pattern
# need good system for notating hand types
# AS 3S 5S 7S 9S JS has wavelength 2 and suit pattern X
# KS AH 2S 3H 4S 5H 6S 7H 8S 9H TS JH has wavelength 1 and suit pattern XY
# allowed suit patterns are X and XY, for non-combined hands

def get_game_value(card):
    n = card.number
    if n == 13:
        return 0
    elif n == 12:
        return None
    else:
        return n

def get_sort_key(card):
    return (get_game_value(card), get_suit_number(card.suit))

def is_queen(card):
    return card.value == "Q"

def sort_hand(hand):
    hand = sorted(hand, key=get_sort_key)
    if any(c.str == "QS" for c in hand):
        raise Exception("The Queen herself!")
    elif any(is_queen(c) for c in hand):
        raise Exception("An agent of Eris!")
    return hand

def get_suit_number(s):
    return Card.Card.SUITS.index(s)

def get_suit_relation(s1, s2):
    # symmetric
    n1 = get_suit_number(s1)
    n2 = get_suit_number(s2)
    m = [
        "XMOC",
        "MXCO",
        "OCXM",
        "COMX",
    ]
    return m[n1][n2]

def matches_suit_pattern(hand, wavelength, suit_pattern):
    hand = sort_hand(hand)
    game_values = np.array([get_game_value(c) for c in hand])
    diffs = game_values[1:] - game_values[:-1]
    suit_codes = []
    # X is same suit, M is co-majority, C is co-color, O is opposite
    reference_suit = hand[0].suit
    for card in hand:
        relation = get_suit_relation(reference_suit, card.suit)
        suit_codes.append(relation)
    raise # TODO finish this


deck = Card.DeckOfCards()
deck.shuffle()

