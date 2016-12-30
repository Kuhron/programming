import math
import random
import statistics as stats

def c(l):
    return random.choice(l)

def r(k,k2 = None):
    if k2 == None:
        return c(range(k))
    else:
        return c(range(k,k2))

def rn(p):
    return sum([r(10)*(10**i) for i in range(p)])

# initialize price as a random 4-digit number
p_0 = rn(4)
# tick size is 0.005, 0.5, 5, or 50
ts = int(10**math.floor(math.log(p_0,10))/20.0)

n_prices = 25
possible_prices = [p_0 + ts*i for i in range(n_prices)]
extended_possible_prices = [p_0 + ts*i/2.0 for i in range(n_prices*2)]

orders = []
while len(orders) < 20:
    b,a = c(possible_prices),c(possible_prices)
    if b > a:
        _ = b
        b = a
        a = _
    if b != a:
        orders.append({"bs":r(1,11),"b":b,"a":a,"as":r(1,11)})

price_summaries = {p:{"bs":0,"as":0} for p in extended_possible_prices}
for o in orders:
    price_summaries[o["b"]]["bs"] += o["bs"]
    price_summaries[o["a"]]["as"] += o["as"]

# print out the bid and ask sizes at prices that have them
for p in possible_prices:
    if price_summaries[p]["bs"] != 0 or price_summaries[p]["as"] != 0:
        print(price_summaries[p]["bs"],p,price_summaries[p]["as"])
print()

# print out the total buyers and sellers for every price, along with the surplus (<- buyers - sellers)
for p_i in range(len(extended_possible_prices)):
    p = extended_possible_prices[p_i]
    b_t = sum([price_summaries[p_]["bs"] for p_ in extended_possible_prices[p_i:]]) # buyers at prices above or equal to p
    a_t = sum([price_summaries[p_]["as"] for p_ in extended_possible_prices[:p_i+1]]) # sellers at prices below or equal to p
    s = b_t - a_t
    price_summaries[p]["b_t"] = b_t
    price_summaries[p]["a_t"] = a_t
    price_summaries[p]["s"] = s
    print(p,b_t,a_t,s)
print()

best_surplus = float("inf")
best_prices = []
for p in extended_possible_prices:
    s_ = price_summaries[p]["s"]
    if abs(s_) < abs(best_surplus):
        best_prices = [p]
        best_surplus = s_
    elif abs(s_) == abs(best_surplus):
        best_prices.append(p)

print(best_prices)

if best_surplus == 0:
    market_price = stats.mean(best_prices)
elif best_surplus > 0: # buying pressure
    market_price = max(best_prices)
else: # selling pressure
    market_price = min(best_prices)

print(market_price)












