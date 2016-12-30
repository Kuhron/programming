import datetime
import os
import random
import string

from collections import Counter

import numpy as np
import Exchange.BlackScholes as bsm


EXPIRATION_FORMAT = "%Y%m%d:%H%M%S"
UNDERLYING_CHARS = string.ascii_uppercase + "0123456789" + "_"


def get_filepath_from_underlying(underlying):
    return "Exchange/data-{}.txt".format(underlying)


def underlying_has_data(underlying):
    filepath = get_filepath_from_underlying(underlying)
    return os.path.exists(filepath)


def verify_underlying(underlying):
    assert all(x in UNDERLYING_CHARS for x in underlying)


def parse_line(line):
    time_str, val_str = line.split(",")
    t = float(time_str)
    val = float(val_str)
    return t, val


def get_settlement_value(underlying, settlement_time):
    filepath = get_filepath_from_underlying(underlying)
    with open(filepath) as f:
        lines = f.readlines()

    before_val = None
    after_val = None
    still_before = True
    for line in lines:
        t, val = parse_line(line)
        if t < settlement_time:
            before_val = val
        else:
            if not still_before:  # flag set already
                after_val = val
                break
            else:
                still_before = False

    if before_val is None or after_val is None:
        return None
    return (before_val + after_val) / 2


def is_settled(underlying, settlement_time):
    return get_settlement_value(underlying, settlement_time) is not None


def get_spot(underlying):
    filepath = get_filepath_from_underlying(underlying)
    with open(filepath) as f:
        lines = f.readlines()

    last_line = lines[-1]
    t, val = parse_line(last_line)
    return val


def get_vol(underlying):
    filepath = get_filepath_from_underlying(underlying)
    with open(filepath) as f:
        lines = f.readlines()

    times = []
    samples = []
    for line in lines:
        if random.random() < 0.01:
            t, val = parse_line(line)
            times.append(t)
            samples.append(val)

    times = times[-100:]
    samples = samples[-100:]

    if len(samples) < 2:
        return random.random()

    times = np.array(times)
    del_ts = times[1:] - times[:-1]
    average_del_t = np.mean(del_ts)
    std = np.std(samples)
    vol = std / (average_del_t ** 0.5)

    return vol


def get_expiration_string(expiration_dt):
    return expiration_dt.strftime(EXPIRATION_FORMAT)


def get_expiration_dt(expiration_string):
    return datetime.datetime.strptime(expiration_string, EXPIRATION_FORMAT)


def get_contract_from_feedcode(feedcode):
    underlying, contract_type, expiration_string = feedcode.split("_")

    expiration_dt = get_expiration_dt(expiration_string)

    if ":" in contract_type:
        contract_type, *strikes = contract_type.split(":")
        strikes = [int(strike) for strike in strikes]
    else:
        strike = None
        assert contract_type == "F"

    multiplier = 1
    if contract_type == "F":
        return Future(underlying, multiplier, expiration_dt)
    elif contract_type == "C":
        return CallOption(underlying, multiplier, strikes, expiration_dt)
    elif contract_type == "P":
        return PutOption(underlying, multiplier, strikes, expiration_dt)
    elif contract_type == "BC":
        return BinaryCallOption(underlying, multiplier, strikes, expiration_dt)
    elif contract_type == "BP":
        return BinaryPutOption(underlying, multiplier, strikes, expiration_dt)
    elif contract_type == "SD":
        return Straddle(underlying, multiplier, strikes, expiration_dt)
    elif contract_type == "SG":
        return Strangle(underlying, multiplier, strikes, expiration_dt)
    elif contract_type == "CS":
        return CallSpread(underlying, multiplier, strikes, expiration_dt)
    elif contract_type == "PS":
        return PutSpread(underlying, multiplier, strikes, expiration_dt)
    elif contract_type == "CF":
        return CallFly(underlying, multiplier, strikes, expiration_dt)
    elif contract_type == "PF":
        return PutFly(underlying, multiplier, strikes, expiration_dt)
    else:
        raise Exception("unknown contract type {}".format(contract_type))


class Contract:
    def is_settled(self):
        if hasattr(self, "already_settled"):
            return True
        else:
            result = is_settled(self.underlying, self.expiration_dt.timestamp())
            if result:
                self.already_settled = True
                self.settlement_value = get_settlement_value(self.underlying, self.expiration_dt.timestamp())
                return True
            else:
                return False

    def get_settlement_value(self):
        if hasattr(self, "settlement_value"):
            return self.settlement_value
        return None


class Future(Contract):
    def __init__(self, underlying, multiplier, expiration_dt):
        self.underlying = underlying
        self.multiplier = multiplier
        self.expiration_dt = expiration_dt
        self.feedcode = self.underlying + "_F_" + get_expiration_string(self.expiration_dt)

    def raw_payoff(self, spot):
        return spot

    def payoff(self, spot):
        return self.multiplier * self.raw_payoff(spot)

    def theo(self, spot):
        return spot

    def greeks(self, spot, v, now):
        return Counter({"price": spot, "delta": 1})


class Option(Contract):
    def __init__(self, contract_type_code, underlying, multiplier, strikes, expiration_dt):
        self.underlying = underlying
        self.multiplier = multiplier
        assert len(strikes) == 1
        self.strike = strikes[0]
        self.expiration_dt = expiration_dt
        self.feedcode = self.underlying + "_{}:{}_".format(contract_type_code, self.strike) + get_expiration_string(self.expiration_dt)

    def payoff(self, spot):
        return self.multiplier * self.raw_payoff(spot)

    def theo(self, spot, v, now):
        greeks = self.greeks(spot, v, now)
        return greeks["price"]


class CallOption(Option):
    def __init__(self, underlying, multiplier, strikes, expiration_dt):
        super().__init__("C", underlying, multiplier, strikes, expiration_dt)

    def raw_payoff(self, spot):
        return max(0, spot - self.strike)

    def greeks(self, spot, v, years_left):
        S = spot
        K = self.strike
        T = max(0, years_left)
        r = 0
        d = 0
        return Counter(bsm.black_scholes("c", S, K, T, r, d, v))


class PutOption(Option):
    def __init__(self, underlying, multiplier, strikes, expiration_dt):
        super().__init__("P", underlying, multiplier, strikes, expiration_dt)

    def raw_payoff(self, spot):
        return max(0, self.strike - spot)

    def greeks(self, spot, v, years_left):
        S = spot
        K = self.strike
        T = max(0, years_left)
        r = 0
        d = 0
        return Counter(bsm.black_scholes("p", S, K, T, r, d, v))


class BinaryCallOption(Option):
    def __init__(self, underlying, multiplier, strikes, expiration_dt):
        super().__init__("BC", underlying, multiplier, strikes, expiration_dt)

    def raw_payoff(self, spot):
        return 1 if spot > self.strike else 0

    # for theo and greeks, treat as N call spreads 1/N strikes apart (or whatever)
    def greeks(self, spot, v, now):
        strike_dist = 0.5
        o1 = CallOption(self.underlying, self.multiplier, self.strike - strike_dist, self.expiration_dt)
        o2 = CallOption(self.underlying, self.multiplier, self.strike + strike_dist, self.expiration_dt)
        g1 = o1.greeks(spot, v, now)
        g2 = o2.greeks(spot, v, now)
        keys = g1.keys()
        assert keys == g2.keys()
        return Counter({k: g1[k] - g2[k] for k in keys})


class BinaryPutOption(Option):
    def __init__(self, underlying, multiplier, strikes, expiration_dt):
        super().__init__("BP", underlying, multiplier, strikes, expiration_dt)

    def raw_payoff(self, spot):
        return 1 if spot < self.strike else 0

    # for theo and greeks, treat as N put spreads 1/N strikes apart (or whatever)
    def greeks(self, spot, v, now):
        strike_dist = 0.5
        o1 = PutOption(self.underlying, self.multiplier, self.strike + strike_dist, self.expiration_dt)
        o2 = PutOption(self.underlying, self.multiplier, self.strike - strike_dist, self.expiration_dt)
        g1 = o1.greeks(spot, v, now)
        g2 = o2.greeks(spot, v, now)
        keys = g1.keys()
        assert keys == g2.keys()
        return Counter({k: g1[k] - g2[k] for k in keys})


class LinearCombinationDerivative(Contract):
    def __init__(self, contract_class_to_strike_index_tuple_dict, contract_type_code, underlying, multiplier, strikes, expiration_dt):
        self.d = contract_class_to_strike_index_tuple_dict
        self.underlying = underlying
        self.multiplier = multiplier
        self.strikes = strikes
        self.expiration_dt = expiration_dt
        strike_str = ":".join(str(k) for k in self.strikes)
        self.feedcode = self.underlying + "_{}:{}_".format(contract_type_code, strike_str) + get_expiration_string(self.expiration_dt)
        self.legs = self.create_legs()

    def create_legs(self):
        legs = []
        for contract_class, strike_index_list in self.d.items():
            for strike_index in strike_index_list:
                strike = self.strikes[strike_index]
                legs.append(contract_class(self.underlying, self.multiplier, (strike,), self.expiration_dt))
        return legs

    def raw_payoff(self, spot):
        return sum(leg.raw_payoff(spot) for leg in self.legs)

    def payoff(self, spot):
        return self.multiplier * self.raw_payoff(spot)

    def greeks(self, spot, v, now):
        result = Counter()
        for leg in self.legs:
            result += Counter(leg.greeks(spot, v, now))
        return result

    def theo(self, spot, v, now):
        greeks = self.greeks(spot, v, now)
        return greeks["price"]


class Straddle(LinearCombinationDerivative):
    def __init__(self, underlying, multiplier, strikes, expiration_dt):
        d = {
            PutOption: {0: 1},
            CallOption: {0: 1},
        }
        super().__init__(d, "SD", underlying, multiplier, strikes, expiration_dt)


class Strangle(LinearCombinationDerivative):
    def __init__(self, underlying, multiplier, strikes, expiration_dt):
        d = {
            PutOption: {0: 1},
            CallOption: {1: 1},
        }
        super().__init__(d, "SG", underlying, multiplier, strikes, expiration_dt)


class CallSpread(LinearCombinationDerivative):
    def __init__(self, underlying, multiplier, strikes, expiration_dt):
        d = {
            CallOption: {0: 1, 1: -1},
        }
        super().__init__(d, "CS", underlying, multiplier, strikes, expiration_dt)


class PutSpread(LinearCombinationDerivative):
    def __init__(self, underlying, multiplier, strikes, expiration_dt):
        d = {
            PutOption: {0: -1, 1: 1},
        }
        super().__init__(d, "PS", underlying, multiplier, strikes, expiration_dt)


# class CallFly(LinearCombinationDerivative):
#     ? #CF


# class PutFly(LinearCombinationDerivative):
#     ? #PF
