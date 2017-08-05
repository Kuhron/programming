import random

cards = [str(i) for i in range(2,11)] + ["J","Q","K","A"]
simple_values = dict(zip(cards,[i for i in range(2,11)]+[10]*3+[1])) # excludes A = 11, but this is accounted for in the value_hand function

# assume infinite well of all card values, draw from uniform distribution with replacement
# because of repeated values, hands should be lists rather than sets

def value_hand(lst):
    if len(lst) == 0:
        return 0
    low_val = sum([simple_values[k] for k in lst])
    ace_count = lst.count("A")
    # for each ace that is used as 11, just add 10 to the value gotten by making them all 1
    cands = [low_val + 10*i for i in range(ace_count+1)]
    valid_cands = [i for i in filter(lambda x: x <= 21, cands)]
    # if there is no way to get 21 or less, return the minimum candidate
    if len(valid_cands) == 0: # len(cands) should never be 0
        return min(cands)
    return max(valid_cands)

def get_cards(n):
    return [random.choice(cards) for i in range(n)]

class Player:
    def __init__(self):
        self.hand = []

    def deal(self):
        self.hand += get_cards(2)

    def hit(self):
        self.hand += get_cards(1)

    def points(self):
        return value_hand(self.hand)

    def has_busted(self):
        return self.points() > 21

    def reset(self):
        self.hand = []

    def busts_this_time(self, cutoff):
        self.deal()
        while self.points() < cutoff:
            self.hit()
        b = self.has_busted()
        self.reset()
        return b

class Dealer(Player):
    # should inherit from Player
    # The difference will be the fact that only one card is visible while players are still playing
    # and that the dealer must soft hit at 17 (or whatever it is)
    def __init__(self):
        self.hand = []

P1 = Player()
D = Player() # make Dealer later

def not_bust_percentage(cutoff):
    busts = 0
    tries = 0
    for i in range(10**5):
        if P1.busts_this_time(cutoff):
            busts += 1
        tries += 1
    bust_perc = 100.0*float(busts)/tries
    #print("Busted {0}% of the time.".format(bust_perc))
    return 100-bust_perc

def win_vs_dealer_percentage(cutoff):
    wins = 0
    tries = 0
    for i in range(10**5):
        D.deal()
        P1.deal()
        while P1.points() < cutoff:
            P1.hit()
        while D.points() < 17:
            # ignores soft hit thing, just uses simple cutoff of 17
            D.hit()
        if (not P1.has_busted()) and (P1.points() > D.points()):
            # ignores possibility of a "push", when you tie the dealer, and you get your money back with no gain or loss from the round
            wins += 1
        D.reset()
        P1.reset()
        tries += 1
    return 100*float(wins)/tries

import matplotlib.pyplot as plt
x = [i for i in range(22)]
win_percs = [win_vs_dealer_percentage(i) for i in range(22)]
plt.plot(x,win_percs)
plt.show()