import math, random

def change(price):
    return price + random.normalvariate(0, price/30.0)

def prompt():
    o = input("Order: ")
    if o == "":
        return 0
    else:
        return int(o)

def next(old_price, price, shares, p_and_l, strategy, verbose = True):
    diff = price - old_price
    if strategy == "user":
        if verbose:
            print("p: {0:.3f}".format(price))
        order = prompt()
    elif strategy == "trend":
        order = int(100*(diff/abs(diff)))
    elif strategy == "contrarian":
        order = int(-100*(diff/abs(diff)))
    shares += order
    p_and_l -= (order*price)
    if verbose:
        print("s: {0}; pl: {1:.2f}".format(shares, p_and_l))
    return shares, p_and_l

if __name__ == "__main__":
    p = 100#10.0**(2.5*random.random()+0.2)
    print(p)
    s_trend = 0 # shares held
    s_contrarian = 0
    pl_trend = 0.0 # p&l
    pl_contrarian = 0.0

    for i in range(10**6):
        #print()
        _p = p
        p = change(p)
        #print(p)
        s_trend, pl_trend = next(_p, p, s_trend, pl_trend, "trend", verbose = False)
        s_contrarian, pl_contrarian = next(_p, p, s_contrarian, pl_contrarian, "contrarian", verbose = False)
    print("trend:", pl_trend)
    print("contrarian", pl_contrarian)

