import math
import random
import sys

def gen_value():
    """
    Defines the target intrinsic value; min 1, max 1 trillion.
    """
    return 10**(1.5+random.random()*2)

def CI(value):
    """
    Uses a base larger than 10 to estimate the order of magnitude
    (crudely, due to the base's size) and returns a confidence interval
    that definitely contains the value.
    """
    base = 10+(random.random()*10)
    order_of_magnitude = math.log(value, base)
    low_est = base ** math.floor(order_of_magnitude)
    high_est = base ** math.ceil(order_of_magnitude)
    return [low_est, high_est]

def get_market():
    """
    Prompts the user for a bid and an ask, returns both."
    """
    ba = input().split("+")
    bid = ba[0]
    try:
        ask = ba[1]
    except:
        print("There is something wrong with the market you gave!")
        return -1, -1
    try:
        int(bid)
    except:
        print("There is something wrong with the bid!")
        return -1, -1
    try:
        int(ask)
    except:
        print("There is something wrong with the ask!")
        return -1, -1
    return int(bid), int(ask)

def trade(value, position, pl, liquidate = False):
    supposed_value = max(0, random.normalvariate(value, value/10))
    if liquidate:
        bid = -1
        ask = -1
        bid_size = abs(position)
        ask_size = abs(position)
    else:
        bid = -1
        ask = -1
        bid_size = -1
        ask_size = -1
    while bid == -1 or ask == -1:
        print("What's your market? Format: bid+ask")
        bid, ask = get_market()
        #print(bid, ask)
    if liquidate:
        if position < 0:
            bid = max(bid, supposed_value)
            position += bid_size
            pl -= bid * bid_size
            print("Sold at {2}! You now have {0} contracts. "
                  #"Your P&L is {1}."
                  .format(position, pl, bid))
            return position, pl
        elif position > 0:
            ask = min(ask, supposed_value)
            position -= ask_size
            pl += ask * ask_size
            print("Bought at {2}! You now have {0} contracts. "
                  #"Your P&L is {1}."
                  .format(position, pl, ask))
            return position, pl
    else:
        while bid_size == -1 or ask_size == -1:
            print("What's your size? Format: bid_size+ask_size")
            bid_size, ask_size = get_market()
            #print(bid_size, ask_size)
        if bid >= supposed_value and ask >= supposed_value:
            position += bid_size
            pl -= bid * bid_size
            print("Sold at {2}! You now have {0} contracts. "
                  #"Your P&L is {1}."
                  .format(position, pl, bid))
        elif ask <= supposed_value and bid <= supposed_value:
            position -= ask_size
            pl += ask * ask_size
            print("Bought at {2}! You now have {0} contracts. "
                  #"Your P&L is {1}."
                  .format(position, pl, ask))
        else:
            print("No trade. You have {0} contracts. "
                  #"Your P&L is {1}."
                  .format(position, pl))
        return position, pl

"""
def liquidate(value, position, pl):
    print("What's your market? Format: bid+ask")
    bid, ask = get_market()
    print(bid, ask)
    print("What's your size? Format: bid_size+ask_size")
    bid_size = math.abs(position)
    ask_size = math.abs(position)
    print(bid_size, ask_size)
"""

if __name__ == "__main__":
    value = gen_value()
    #print(value)
    ci = CI(value)
    if value < ci[0] or value > ci[1]:
        print("There is something wrong with CI(value)!")
        sys.exit()
    position = 0
    num_trades = 0
    min_trades = 3
    max_trades = 100
    pl = 0
    print("You start with {0} contracts.".format(position))
    print("Confidence interval: ", ci)
    while num_trades < min_trades:
        position, pl = trade(value, position, pl)
        num_trades += 1
    while position != 0 and num_trades < max_trades:
        position, pl = trade(value, position, pl)
        num_trades += 1
    #print("Time to liquidate your position.")
    #liquidate(value, position, pl)
    if position == 0:
        print("You have liquidated your position.\n"
              "The value was {2:.2f}.\n"
              "Your P&L is {0} ({1:.2f}%; avg. {3:.2f}% per trade)."
              .format(pl, 100*pl/value, value, 100*pl/(value*num_trades)))
    else:
        print("You have reached the maximum number of trades. Liquidating position.")
        position, pl = trade(value, position, pl, liquidate = True)
        print("The value was {2:.2f}.\n"
              "Your P&L is {0} ({1:.2f}%; avg. {3:.2f}% per trade)."
              .format(pl, 100*pl/value, value, 100*pl/(value*num_trades)))













