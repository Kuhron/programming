import random

import numpy as np
import matplotlib.pyplot as plt

def draw_addend():
    """
    Returns 0 1/2 of the time, 1 1/4 of the time, 2 1/8 of the time, etc.
    """
    # # confusing version
    # r = random.random()
    # if r >= 0.5:
    #     return 0
    # # this loop MUST return at some point because r cannot be zero
    # i = 0
    # while r < 2.0**(-i): # simple case: r = 0.4, so r < 2**-1 is true, i goes to 2, and 2**0 is returned
    #     i += 1
    # return 2**(i-2)

    # # easier version
    addend = 0
    if random.random() < 0.5:
        addend = 1
    while random.random() < 0.5:
        addend *= 2
    return addend

# a = []
# for i in range(10**5):
#     a.append(draw_addend())
# print(0, a.count(0)/float(len(a)))
# for i in range(10):
#     print(2**i, a.count(2**i)/float(len(a)))

def extend_wait(remaining_wait):
    return remaining_wait + draw_addend()

def black_swan_occurs(remaining_wait):
    return remaining_wait <= 0.0 # less than should never happen, but just in case

# a = []
# starting_wait = 2
# wait = starting_wait
# for t in range(10**6):
#   wait -= 1
#   wait = extend_wait(wait)
#   #print(wait)
#   if black_swan_occurs(wait):
#       a.append(t)
#       wait = starting_wait
# print(a)

# try the "fairly select an element from a list of unknown length" method from the Wolverine interview

# this tells us the minimum float, which is 2**-1075 (this prints 1076)
# p = 10
# min_float = 2.0**-p
# while min_float != 0:
#     p += 1
#     min_float = 2.0**-p
# print(p)

def get_random_integer(precision, signed = True):
    """
    precision is the amount by which the list length is multiplied with probability 1/precision.
    smaller precision gives better variation for list lengths but takes longer
    """
    # # one way
    # cand = 1
    # selection = 1
    # while cand < 10**2:
    #     cand += 1
    #     if random.random() < 1.0/cand:
    #         selection = cand
    # return selection

    # another way
    length = 1
    while random.random() < 1.0/precision:
        length *= precision
    c = random.choice([i for i in range(int(length))])
    if signed and c != 0 and random.random() < 0.5: # does this introduce bias for or against returning 0? who cares!
        c = -c
    return c

# volume = []
# price = [0]
# for i in range(10**6):
#     number = get_random_integer(2)
#     volume.append(abs(number))
#     price.append(price[-1] + number)
# #print([b for b in filter(lambda x: x > 100, a)])
# plt.plot(volume)
# plt.show()
# plt.plot(price)
# plt.show()

def play_market(money, periods, price, ui = True):
    #print(sorted(a))
    if ui:
        print("Welcome to the market. Here you can buy tickets with infinite expected payout, but at a price. The current price is {0}.".format(price))
    i = 0
    while i < periods:
        pay = get_random_integer(100, signed = False)
        if ui:
            tickets_raw = input("You have ${0:.0f}. How many lots would you like to buy? (default 1) ".format(money))
            if tickets_raw == "":
                tickets = 1
            else:
                tickets = int(tickets_raw)
        else:
            tickets = 1
        money -= tickets*price # cost
        if ui:
            print("The tickets paid ${0} each at period {1}.".format(pay,i))
        money += tickets*pay # payout
        if money <= 0:
            if ui:
                print("You ran out of money.")
            break
        i += 1
    return i # number of periods survived

#play_market(money=100, periods=1000, price=1)

def show_average_evolution(periods, precision, signed = True):
    t = 0.0
    avgs = []
    for i in range(1,periods+1):
        t += get_random_integer(precision, signed)
        avgs.append(t/i)
        #print("Average = {0} at period {1}".format(t/i,i))
    plt.plot(avgs)
    plt.show()

#show_average_evolution(100000,100,signed=False)

def show_survival_times(trials, money, periods, price):
    v = []
    for i in range(trials):
        v.append(play_market(money,periods,price,ui = False))
    plt.plot(v)
    plt.show()
    return sum(v)/float(trials)

#print(show_survival_times(trials=1000,money=100,periods=10**7,price=1))

# with pareto distribution
def pareto_f(k,alpha=1):
    a = [np.random.pareto(alpha)*random.choice([1,-1]) for i in range(k)]
    b = [a[0]]
    for i in range(1,k):
        b.append(b[i-1]+a[i])
    return a,b

def p(x):
    plt.plot(x)
    plt.show()

def pareto_g(k,alpha=1):
    a,b = pareto_f(k,alpha)
    p(a)
    p(b)

pareto_g(10**6,alpha=1)






