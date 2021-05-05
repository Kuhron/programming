# idea: have a predictable component like seasonal/climatic temperature average
# and use hash to get deterministic random variation around those averages
# see what co-occurs on different dimensions on the same day,
# e.g. atmospheric/seismic/etc.
# can also have seasonal variation in the probability of something
# and the hash represents rolling the dice to see if that probability is met


from hashlib import sha256
from datetime import datetime
import time
import random


def get_hash():
    # use current time, or later can write function with some universal Cadan time format
    dt = datetime.utcnow().strftime("%Y%m%d-%H%M%S").encode("utf-8")
    b = sha256(dt).digest()
    n = int.from_bytes(b, "big")
    return n


def hash_choice(options):
    n = get_hash()
    i = n % len(options)
    return options[i]


def hash_if(probability):
    max_n = 2**256
    n = get_hash()
    r = n/max_n
    return r < probability


if __name__ == "__main__":
    # todo: introduce seasonal variation and variation by place
    # todo: joint distribution, events like thunderstorm and tornado are not independent
    options = [
        ["rain", 0.1], ["cloudy", 0.3], ["sunny", 0.6], ["thunderstorm", 0.1],
        ["flood", 0.02], ["tornado", 0.01], ["volcano", 0.001], ["hurricane", 0.005],
        ["earthquake", 0.0001], ["dust", 0.08], ["wildfire", 0.03], ["tsunami", 0.001],
        ["ashfall", 0.002], ["fog", 0.02], ["steam", 0.04],
    ]
    for day in range(100):
        print(f"Day {day}")
        weather_events = []
        for option, p in options:
            if hash_if(p):
                weather_events.append(option)
        print("weather events today:", weather_events)
        time.sleep(1)
