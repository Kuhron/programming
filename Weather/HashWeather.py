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
import math
import numpy as np
import matplotlib.pyplot as plt


MAX_HASH = 2**256

def get_dt_str():
    # use current time, or later can write function with some universal Cadan time format
    dt = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return dt


def get_hash(s):
    byts = s.encode("utf-8")
    b = sha256(byts).digest()
    n = int.from_bytes(b, "big")
    return n


def hash_choice(options, dt_str):
    n = get_hash(dt_str)
    i = n % len(options)
    return options[i]


def hash_if(probability, dt_str, event_name_salt):
    r = hash_random_01(dt_str + event_name_salt)  # so that e.g. r comes out as 0.01 doesn't mean EVERY event with probability greater than that happens all at once
    return r < probability


def hash_random_01(obj):
    s = str(obj)
    n = get_hash(s)
    return n / MAX_HASH


def hash_random_sequence_01(obj):
    # generator, deterministic sequence of pseudo-random values seeded by object
    s = str(obj)
    r = ""  # so initial s + r is just s
    while True:
        r = hash_random_01(s + str(r))
        yield r


def hash_random_uniform(obj, a, b):
    r = hash_random_01(obj)
    return a + r * (b-a)


def hash_random_sequence_uniform(obj, a, b):
    seq = hash_random_sequence_01(obj)
    for r in seq:
        yield a + r * (b-a)


def test_hash_random_sequence_01():
    # should look random, mean near 0.5, no autocorrelation, etc.
    x = random.uniform(0, 100)
    seq = hash_random_sequence_01(x)
    ys = [next(seq) for i in range(10000)]
    print(f"mean value of sequence is {np.mean(ys)}")
    for offset in range(1, 6):
        autocorrelation = np.corrcoef(ys[:-offset], ys[offset:])[0,1]
        print(f"autocorrelation offset={offset}: {autocorrelation}")
    plt.scatter(range(len(ys)), ys)
    plt.show()


def get_hash_wave_function(seed_object):
    # something like: for each power of 2 in the binary search, that power of 2 is a "fencepost", a pinned value, and we add noise between those at smaller and smaller precisions
    # so you get big fluctuations over longer timescales, and smaller amounts of noise on smaller timescales around that long-term trend
    # but the "spectrum" can be changed probably, e.g. you don't even have to reduce the noise on small timescales (completely white noise), and it would result in a function that in the limit is not continuous (could jump to any value anywhere)
    # or could just change the exponent of decay of the amplitude of waves by their frequency, to get smoother/rougher noise (corresponding to spectra with less/more energy at higher frequencies)
    raise NotImplementedError


def get_binary_search_sequence(number):
    # generator
    # sequence of, for each power of 2, tuple (power, which multiples of that power of 2 this number is between)
    # for powers higher than the number, will get (power, 0, 1*2**power) for all of them
    # treat negatives symmetrically to positives

    x = float(number)
    if x == 0:
        power = 0
        while True:
            tup = (power, 0, 2**power)
            yield tup
            power -= 1
    # rest should be unreachable for 0 because of while loop

    sign = np.sign(x)
    x /= sign
    flog2 = math.floor(np.log2(x))
    clog2 = math.ceil(np.log2(x))
    starting_left_number = 2 ** flog2
    starting_right_number = 2 ** clog2
    starting_power = flog2  # the power is the exponent, not 2**exponent

    if flog2 == clog2:
        # x is a power of 2, all the binary items will just be (power, x, x)
        assert 2**flog2 == x and 2**clog2 == x
        power = starting_power
        while True:
            tup = (power, sign*x, sign*x)
            yield tup
            power -= 1
    # rest should be unreachable for powers of 2

    assert starting_left_number == 2 ** starting_power
    assert starting_right_number == 2 * (2 ** starting_power)
    assert starting_right_number == 2 * starting_left_number, "math error"
    # 2**flog2 and 2**clog2 are both multiples of what power? it must be 2**flog2 (they are 1 and 2 times that number)

    left_number = starting_left_number
    right_number = starting_right_number
    power = starting_power
    while True:
        tup = (power, sign*left_number, sign*right_number)
        yield tup

        average_number = (left_number + right_number)/2
        left_distance = abs(x - left_number)
        right_distance = abs(x - right_number)
        # print(average_number, left_distance, right_distance)
        if left_distance == right_distance:
            # it's halfway between them, so the next power will be the average number, because that should be equal to the target number, so all subsequent powers will be the same number
            assert x == average_number
            left_number = average_number
            right_number = average_number
            # this should persist forever because the average will always stay the same now
        elif left_number < x < average_number:
            # left number stays the same
            right_number = average_number
        elif average_number < x < right_number:
            left_number = average_number
            # right number stays the same
        else:
            raise RuntimeError(f"problem with binary search for x={x}")
        power -= 1


def unit_test_binary_search():
    binary_search_test_cases = [0, 2, 4096, 64*3, -256*7, -2, -16384, 7777.7777] + [random.uniform(-10**9, 10**9) for i in range(100)]
    for x in binary_search_test_cases:
        print(x)
        binary_search = get_binary_search_sequence(x)
        for tup in binary_search:
            print(tup)
            if x >= 0:
                assert tup[1] <= x <= tup[2]
            else:
                assert tup[2] <= x <= tup[1]  # bigger magnitude is tup[2]
            if tup[1] == 0:
                # all non-zero numbers should be between SOME power of two and two times that number
                assert x == 0
            if tup[0] < -5:
                break


def run_simple_simulation():
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



if __name__ == "__main__":
    # todo: introduce seasonal variation and variation by place
    # todo: joint distribution, events like thunderstorm and tornado are not independent
    # run_simple_simulation()

    # want to get a deterministic function with good autocorrelation like long-term variations, and smaller variations around that trend, etc. to fractal precision

    seed = random.uniform(0, 100)
    print(f"seed is {seed}")

    hash_wave_function = get_hash_wave_function(seed_object=seed)

    xs = np.arange(0, 100, 0.01)
    ys = hash_wave_function(xs)
