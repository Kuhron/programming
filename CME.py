import collections
import random

import numpy as np


class Instrument:
    def __init__(self):
        pass


class Outright(Instrument):
    def __init__(self, symbol):
        self.underlying = symbol[:2]
        self.expiry = symbol[2:4]


class Roll(Instrument):
    def __init__(self, symbol):
        self.leg_symbols = symbol.split("-")
        self.legs = [Outright(leg_symbol) for leg_symbol in leg_symbols]
        self.price = self.get_price()

    def get_price(self):
        p0 = self.legs[0].get_price()
        p1 = self.legs[1].get_price()
        return p0 - p1


class OrderBookEntry:
    def __init__(self):
        pass


class OrderBook:
    def __init__(self,instrument):
        self.buys = collections.defaultdict(list)
        self.sells = collections.defaultdict(list)
        self.book = {"buy":self.buys,"sell":self.sells}

    def submit_order(self, order):
        size_to_be_executed = min(order.size, sum(self.book[order.other_side][order.price]))
        self.execute_order_fifo(order)
        self.book[order.side][order.price].append(order.size - size_to_be_executed)

    def execute_order_fifo(self, order):
        executed_size = 0
        while sum(self.book[order.other_side][order.price]) > 0 and executed_size < order.size:
            next_size_to_execute = min(self.book[order.other_side][order.price][0],order.size-executed_size)
            executed_size += next_size_to_execute
            if next_size_to_execute == self.book[order.other_side][order.price][0]:
                self.book[order.other_side][order.price] = self.book[order.other_side][order.price][1:]
            else:
                self.book[order.other_side][order.price][0] -= next_size_to_execute
        if executed_size > 0:
            print("order to {0} {1} {2} at price {3} was executed for size {4}".format(order.side,order.size,order.instrument,order.price,executed_size))

    def get_prices(self):
        price_set = set()
        price_set |= set(self.buys.keys())
        price_set |= set(self.sells.keys())
        return sorted(price_set, reverse = True)

    def show(self):
        for price in self.get_prices():
            a = str(sum(self.book["buy"][price]))
            b = "{0:.2f}".format(price)
            c = str(sum(self.book["sell"][price]))
            print(a.rjust(6)+b.rjust(6)+c.rjust(6))


class Order:
    def __init__(self,side,size,instrument,price):
        self.display_size = float("inf")
        self.side = side
        self.other_side = "buy" if side == "sell" else "sell"
        self.size = size
        self.instrument = instrument
        self.price = price

    def show(self):
        print("order to {0} {1} {2} at price {3:.2f}".format(self.side,self.size,self.instrument,self.price))


order_book = OrderBook("X0FM")

for i in range(1000):
    price = random.choice(np.arange(1,8,0.5))
    side = random.choice(["buy","sell"])
    size = random.choice(range(1,101))
    order_book.submit_order(Order(side,size,"X0FM",price))
    order_book.show()
    waste = input("Press enter to continue ")










