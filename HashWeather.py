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


def get_dt_str():
    # use current time, or later can write function with some universal Cadan time format
    dt = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return dt


def get_hash(dt_str):
    dt_bytes = dt_str.encode("utf-8")
    b = sha256(dt_bytes).digest()
    n = int.from_bytes(b, "big")
    return n


def hash_choice(options, dt_str):
    n = get_hash(dt_str)
    i = n % len(options)
    return options[i]


def hash_if(probability, dt_str, event_name_salt):
    max_n = 2**256
    n = get_hash(dt_str + event_name_salt)  # so that e.g. r comes out as 0.01 doesn't mean EVERY event with probability greater than that happens all at once
    r = n/max_n
    return r < probability


if __name__ == "__main__":
    # todo: introduce seasonal variation and variation by place
    # todo: joint distribution, events like thunderstorm and tornado are not independent
    options = [
        ["rain", 0.1], ["cloudy", 0.3], ["thunderstorm", 0.1], ["snow", 0.0001],
        ["flood", 0.02], ["tornado", 0.01], ["volcano", 0.001], ["hurricane", 0.005],
        ["earthquake", 0.0001], ["dust", 0.08], ["wildfire", 0.03], ["tsunami", 0.001],
        ["ashfall", 0.002], ["fog", 0.02], ["steam", 0.04], ["smoke", 0.05], ["haze", 0.08],
        ["hail", 0.0002], ["blizzard", 0.00005], ["landslide", 0.005], ["rock_hail", 0.002],
        ["lava_flow", 0.001], ["mud_flow", 0.003], ["pollen", 0.009],
    ]
    default_option = "sunny"
    for day in range(100):
        print(f"Day {day}")
        dt_str = get_dt_str()
        weather_events = []
        for option, p in options:
            if hash_if(p, dt_str, event_name_salt=option):
                weather_events.append(option)
        if len(weather_events) == 0:
            weather_events.append(default_option)
        print("weather events today:", weather_events)
        time.sleep(1)
