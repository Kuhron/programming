import datetime
import msvcrt
import random
import time

from collections import Counter

from Exchange.Contracts import get_contract_from_feedcode, get_spot, get_vol


class Market:
    def __init__(self, bid, ask):
        self.bid = bid
        self.ask = ask

    def __repr__(self):
        return "{:6.2f} @ {:6.2f}".format(self.bid, self.ask)


class MarketMaker:
    def __init__(self):
        self.trader = Trader()

    def get_market(self, contract):
        theo = self.get_theo(contract)
        edge = self.get_edge(contract)
        bid = round(theo - edge, 2)
        ask = round(theo + edge, 2)
        return Market(bid, ask)

    @staticmethod
    def get_theo(contract):
        return MarketMaker.get_greeks(contract)["price"]

    @staticmethod
    def get_greeks(contract):
        # return random.random()
        spot = get_spot(contract.underlying)
        N_SECONDS_PER_YEAR = 86400 * 365
        vol = get_vol(contract.underlying)
        years_left = (contract.expiration_dt - datetime.datetime.now()).total_seconds() / N_SECONDS_PER_YEAR
        # if years_left <= 0:
        #     return Counter({"price": contract.payoff(spot)})
        return contract.greeks(spot, vol, years_left)

    def get_edge(self, contract):
        return random.uniform(0.01, 0.1)


class Inventory:
    def __init__(self, feedcode):
        self.feedcode = feedcode
        self.position = 0


class Trader:
    def __init__(self):
        self.inventory = {}
        self.cash_balance = 0

    def trade(self, contract, size, price):
        feedcode = contract.feedcode
        if feedcode not in self.inventory:
            self.inventory[feedcode] = Inventory(feedcode)
        self.inventory[feedcode].position += size
        self.cash_balance += (-1 * size) * contract.multiplier * price

    def expire_positions(self, contract, spot, verbose=False):
        assert spot is not None

        feedcode = contract.feedcode
        if feedcode not in self.inventory:
            return

        pos = self.inventory[feedcode].position
        if pos != 0:
            payoff = contract.payoff(spot)  # already includes multiplier
            self.inventory[feedcode].position = 0
            self.cash_balance += payoff * pos

            if verbose:
                print("Expired {} x {} at {}. Cash {:.2f}".format(pos, feedcode, spot, self.cash_balance))

    def expire_all_positions(self, verbose=False):
        for feedcode in self.inventory:
            contract = get_contract_from_feedcode(feedcode)
            if contract.is_settled():
                spot = contract.get_settlement_value()
                self.expire_positions(contract, spot, verbose=verbose)

    def get_greeks_position(self):
        greeks = Counter()
        for feedcode, inv in self.inventory.items():
            # use the same method as MarketMaker to estimate theo; but will get a different vol estimate
            contract = get_contract_from_feedcode(feedcode)
            new_greeks = MarketMaker.get_greeks(contract)
            # print("greeks for {}: {}".format(feedcode, new_greeks))
            new_greeks_to_add = Counter({k: v * inv.position for k, v in new_greeks.items()})
            # print("your position: {} x {}; greeks {}".format(inv.position, feedcode, new_greeks_to_add))
            # greeks += new_greeks_to_add  # THIS DOES NOT WORK; negative values are deleted
            greeks.update(new_greeks_to_add)  # this produces the intended behavior
        # print("final position: {}".format(greeks))
        return greeks

    def print_position(self):
        # print("Your total position:")
        try:
            pos = self.get_greeks_position()
            s = ""
            for k, v in sorted(pos.items()):
                if k not in ["price", "rho"]:
                    s += "{}: {:.4f}. ".format(k, v)
            print(s)
            print("")
        except ZeroDivisionError:
            print("greeks invalidated due to expired contract")


def buy(contract, user, market_maker, market):
    user.trade(contract, 1, market.ask)
    market_maker.trader.trade(contract, -1, market.ask)
    print("You bought {:.2f}. Position {}. Cash {:.2f}.".format(
        market.ask, user.inventory[feedcode].position, user.cash_balance
    ))
    user.print_position()


def sell(contract, user, market_maker, market):
    user.trade(contract, -1, market.bid)
    market_maker.trader.trade(contract, 1, market.bid)
    print("You sold {:.2f}. Position {}. Cash {:.2f}.".format(
        market.bid, user.inventory[feedcode].position, user.cash_balance
    ))
    user.print_position()


def trade(contract, user, market_maker, market):
    t0 = time.time()
    timeout = 5
    while time.time() - t0 < timeout:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            while key in [b"\000", b"\xe0"]:  # precede special function key codes; ignore them
                key = msvcrt.getch()
            ordinal = ord(key)
            print(key, ordinal)
            if ordinal in [72, 80]: # up and down arrows
                if ordinal == 72:
                    buy(contract, user, market_maker, market)
                    return
                elif ordinal == 80:
                    sell(contract, user, market_maker, market)
                    return
            elif key == b"x":
                print("quitting this contract")
                return "quit"
    return


def trade_contract(contract, user, market_maker):
    print("press up arrow to buy, down arrow to sell, x to quit")
    while not contract.is_settled():
        market = market_maker.get_market(contract)
        print(market)
        user.print_position()
        result = trade(contract, user, market_maker, market)
        if result == "quit":
            return
    if contract.is_settled():
        print("contract has expired")
        spot = contract.get_settlement_value()
        user.expire_positions(contract, spot, verbose=True)
        market_maker.trader.expire_positions(contract, spot)


if __name__ == "__main__":
    user = Trader()
    market_maker = MarketMaker()

    while True:
        user.expire_all_positions(verbose=True)
        market_maker.trader.expire_all_positions()
        feedcode = input("feedcode to trade: ")
        contract = get_contract_from_feedcode(feedcode)
        if contract is None:
            continue
        trade_contract(contract, user, market_maker)