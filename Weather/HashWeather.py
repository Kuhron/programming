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
import functools
import os

import sys
sys.path.insert(0, "/home/wesley/programming")
from InteractivePlot import InteractivePlot


# portability to PyDroid for working on this on planes
android = "ANDROID_BOOTLOGO" in os.environ
if not android:
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

    # let f(0) be 0, otherwise 0 as a fencepost will have infinite contributions from ever-higher powers, giving it infinite variance (when do you stop adding waves from higher powers? the amount they shift the whole function gets bigger and bigger so you can never ignore any of them, so the function can never converge)

    cache = {}  # for the fenceposts at multiples of powers of 2
    def f(x):
        binary_search = get_binary_search_sequence(x)
        

    raise NotImplementedError


def get_test_wave_function_at_fenceposts():
    d = {}
    d[0] = 0 # random.uniform(-1, 1)
    d[100] = 0 # random.uniform(-1, 1)

    # seed_obj = "testingtesting123abcdefg"
    seed_obj = str(time.time())
    iterations = 10
    exponent = 0.5
    for i in range(iterations):
        ks = sorted(d.keys())
        for k0, k1 in zip(ks[:-1], ks[1:]):
            new_k = (k0 + k1) / 2
            mean = (d[k0] + d[k1]) / 2
            amplitude = ((k1 - k0) / 2) ** exponent  # e.g. with keys 0 and 1, we are getting value at 0.5, and want the amplitude to be proportional to that wavelength
            r = hash_random_uniform(seed_obj + str(new_k), -1, 1)  # deterministic random value between -1 and 1, and it's also not dependent on a sequence of random values (where accessing different number of them beforehand can change the result)
            deviation = r * amplitude
            new_val = mean + deviation
            d[new_k] = new_val

    # check evenly spaced (debug)
    ks = sorted(d.keys())
    diff = None
    for k0, k1 in zip(ks[:-1], ks[1:]):
        if diff is None:
            diff = k1 - k0
        else:
            assert k1 - k0 == diff

    return d


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

    sgn = sign(x)
    x /= sgn
    flog2 = math.floor(math.log(x,2))
    clog2 = math.ceil(math.log(x,2))
    starting_left_number = 2 ** flog2
    starting_right_number = 2 ** clog2
    starting_power = flog2  # the power is the exponent, not 2**exponent

    if flog2 == clog2:
        # x is a power of 2, all the binary items will just be (power, x, x)
        assert 2**flog2 == x and 2**clog2 == x
        power = starting_power
        while True:
            tup = (power, sgn*x, sgn*x)
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
        tup = (power, sgn*left_number, sgn*right_number)
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


def get_binary_search_sequence_list(number, min_power=None, n_terms=None):
    assert (min_power is not None) + (n_terms is not None) == 1, "need min_power or n_terms but not both"
    g = get_binary_search_sequence(number)
    if n_terms is not None:
        return [next(g) for i in range(n_terms)]
    elif min_power is not None:
        res = []
        for tup in g:
            power, left, right = tup
            if power >= min_power:
                res.append(tup)
            else:
                return res  # effectively break
    else:
        raise Exception("shouldn't have gotten here because of the assert")


def get_binary_search_fenceposts(number):
    # the list of values whose hash-r11-deviations will be summed to get f(x)
    # note that it's impossible to get fenceposts like A, B, A (e.g. it's closer to 1 than to 0, then it's closer to 1/2 than to 1, but then it's closer to 1 than to 3/4; this can't happen because after it's closer to 1/2 than to 1, the new left/right will be 1/2 and 3/4, and 1 is never again considered as a close number)
    # - as a result of this, we can just check if the number was in the last tuple to determine if it's a duplicate; if it wasn't, then it can't have been in a tuple before that either
    last_val = None
    for tup in get_binary_search_sequence(number):
        power, left_number, right_number = tup
        if number < 0:
            left_number, right_number = right_number, left_number  # they are in the tuple in magnitude order
        dleft = number - left_number
        dright = right_number - number
        assert dleft >= 0 and dright >= 0
        if dleft == dright:
            if dleft == dright == 0:
                assert number == left_number == right_number, f"n={number}\nleft={left_number}\nright={right_number}" 
                if last_val is None:
                    yield number, power
                else:
                    assert last_val == number, f"last={last_val}\nn={number}"  # it should yield the number itself as the last one
                    # don't yield it again
                break  # don't run through the infinite repetitions anymore
            else:
                closer = number
        elif dleft < dright:
            closer = left_number
        elif dright < dleft:
            closer = right_number
        else:
            raise Exception("impossible")
        # if it was yielded in last iteration, it's a duplicate, else, it must be new
        if last_val is not None and closer == last_val:
            # duplicate, don't yield it
            pass
        else:
            yield closer, power
            last_val = closer


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


def r11(x, seed):
    return hash_random_uniform(seed + str(x), -1, 1)


def amp(x, exponent):
    power_of_2 = get_power_of_2_that_number_is_multiple_of(x)
    return power_of_2 ** exponent


def get_power_of_2_that_number_is_multiple_of(x):
    # e.g. for 1/8 and 3/8 return 1/8, for 11/2 return 1/2, for 32*(something coprime to 2) return 32, etc.
    if x == 0:
        return 0
    log2 = math.log(x,2)
    # x can be at most 2**log2
    p = math.ceil(log2)
    lowest_power = -32
    while p >= lowest_power:
        mod = x % (2**p)
        if mod == 0:
            return (2**p)
        p -= 1
    return 0  # x is (approximately) not expressible as q/2**p


@functools.lru_cache(maxsize=100000)
def get_deviation_at_value(x, seed, exponent):
    return r11(x, seed) * amp(abs(x), exponent)


def get_fencepost_deviation_sum(x, seed, exponent):
    res = 0
    for fencepost, power in get_binary_search_fenceposts(x):
        dev = get_deviation_at_value(fencepost, seed, exponent)
        if fencepost == 0:
            assert dev == 0
            continue

        # attempting to get rid of discontinuities by making the contribution of a fencepost continuous within its sphere of influence from 0 (at edge of window) to max (at the fencepost itself), but this is adding some complication that I would rather not have if there's an easier way
        fencepost_window_lower_magnitude_bound = 3/4 * fencepost
        fencepost_window_upper_magnitude_bound = 3/2 * fencepost
        alpha = (x - fencepost_window_lower_magnitude_bound) / (fencepost_window_upper_magnitude_bound - fencepost_window_lower_magnitude_bound)
        # alpha is where x is within the fencepost's sphere of influence
        # note that it still works for negatives, e.g. -1 within [-3, 0] has alpha = (-1 - 0)/(-3 - 0) = -1/-3 = 1/3
        assert 0 <= alpha <= 1
        # at the fencepost itself (alpha = (1-3/4)/(3/2-3/4) = 1/3) we want full deviation to be added, want this to drop off smoothly to 0 at alpha = 0 and 1
        # after a bunch of algebra I have found a family of functions solving this
        scaling = one_third_smooth_bump_function(alpha)
        if abs(alpha - 0) < 1e-6 or abs(alpha - 1) < 1e-6:
            assert abs(scaling - 0) < 1e-6, (alpha, scaling)
        elif abs(alpha - 1/3) < 1e-6:
            assert abs(scaling - 1) < 1e-6, (alpha, scaling)
        else:
            assert 0 < scaling < 1, (alpha, scaling)
        res += dev * scaling
        # print(f"x = {x}, fencepost = {fencepost}, power = {power}, alpha = {alpha}, scaling = {scaling}, dev = {dev}, scaled dev = {dev*scaling}")
    return res


def one_third_smooth_bump_function(x):
    assert 0 <= x <= 1
    # this is a sinusoidal bump function with f(0) = f(1) = 0 and f(1/3) = 1
    
    a = 4  # which period of the sin wave are we at (wanted g(x) to be 0 -> a*pi, 1/3 -> (a+1/2)*pi, 1 -> (a+1)*pi)
    assert type(a) is int  # any int works without changing the result (f)
    b = 12  # the base of the log in unscaled g = log_b(h(x)) : [0,1] -> [0,1], then scaled g : [0,1] -> [a*pi, (a+1)*pi]
    assert b > 1  # otherwise sqrt(b) < 1 and b < sqrt(b) so the h curve doesn't work right

    p = math.log((b-1)/(math.sqrt(b)-1), 3)
    h = 1 + (b-1) * (x**p)
    g = np.pi * (a + math.log(h, b))
    f = np.sin(g)**2
    return f


def test_one_third_smooth_bump_function():
    xs = np.linspace(0, 1, 100)
    ys = np.vectorize(one_third_smooth_bump_function)(xs)
    plt.plot(xs, ys)
    plt.savefig("OneThirdBump.png")


def is_in_window_of_influence(x, fencepost):
    # fencepost shows up in the binary search of x
    # this happens if 3/4 * fencepost <= x <= 3/2 * fencepost
    return 3/4 * fencepost <= x <= 3/2 * fencepost


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


def summarize_xs_ys(xs, ys):
    assert len(xs) == len(ys)
    for i in range(len(xs)-1):
        x0 = xs[i]
        x1 = xs[i+1]
        y0 = ys[i]
        y1 = ys[i+1]
        dx = x1-x0
        dy = y1-y0
        dydx = dy/dx
        print(f"x={x0}\nf(x)={y0}\ndy/dx={dydx}\n")
    x = xs[-1]
    y = ys[-1]
    print(f"x={x}\nf(x)={y}\n")


def print_graph(xs, ys, n_ticks=100):
    f_dict = dict(zip(xs,ys))
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    print(f"{min_x} <= x <= {max_x}")
    print(f"{min_y} <= f(x) <= {max_y}")
    x_tick = (max_x - min_x) / (n_ticks-1)
    print_xs = linspace(min_x, max_x, n_ticks)
    print_ys = []
    for x in print_xs:
        lows = [xx for xx in xs if xx <= x]
        highs = [xx for xx in xs if xx >= x]
        a = max(lows)
        b = min(highs)
        c = f_dict[a]
        d = f_dict[b]
        alpha = (x-a)/(b-a) if b > a else 0
        y = c + alpha * (d-c)
        print_ys.append(y)
    print_y_ticks(print_ys)


def print_y_ticks(ys, min_y=None, max_y=None, print_lines_together=True):
    # min and max default to the min and max of the given ys
    min_y = min_y if min_y is not None else min(ys)
    max_y = max_y if max_y is not None else max(ys)
    if min_y == max_y:
        # will give div/0, so just artificially move them
        min_y -= 1
        max_y += 1

    screen_width = 60
    y_tick = (max_y - min_y) / (screen_width)  # no extra fencepost here, tick is a bin of y vals corresponding to the char width
    assert y_tick > 0, y_tick
    print(f"latest: {ys[-1]}")
    print(f"range: {min_y} - {max_y}")
    s_together = ""
    for y in ys:
        which_y_tick = int((y-min_y) // y_tick)
        if 0 <= which_y_tick <= screen_width:
            # normal case: print default char as the axis and marked char as the approximate y position
            n_y_ticks_before = which_y_tick
            n_y_ticks_after = screen_width - which_y_tick - 1
            s = "." * n_y_ticks_before + "|" + "." * n_y_ticks_after
        elif which_y_tick < 0:
            # out of bounds case
            s = "<<<" + "." * (screen_width-3)
        elif which_y_tick > screen_width:
            s = "." * (screen_width-3) + ">>>"
        else:
            raise Exception("impossible")
        if print_lines_together:
            s_together += "\n" + s
        else:
            print(s)
    if print_lines_together:
        print(s_together)


def print_graph_live(seed, spectrum_exponent, x0, x_step):
    max_rows = (442-414)+1  # measured using Vim line numbers lol
    x = x0
    ys = []
    while True:
        y = get_fencepost_deviation_sum(x, seed, spectrum_exponent)
        ys.append(y)
        ys = ys[-max_rows:]
        os.system("clear")
        print_y_ticks(ys, print_lines_together=True)
        time.sleep(0.01)
        x += x_step


def plot_live(seed, spectrum_exponent, x0, x_step):
    max_frames = 100
    x = x0
    xs = []
    ys = []

    with InteractivePlot(plot_every_n_steps=10) as iplt:
        while iplt.is_open():
            y = get_fencepost_deviation_sum(x, seed, spectrum_exponent)
            xs.append(x)
            ys.append(y)
            xs = xs[-max_frames:]
            ys = ys[-max_frames:]
            x += x_step
            iplt.plot(xs, ys)


    # plt.ion()
    # fignum = plt.gcf().number  # use to determine if user has closed plot
    # while True:
    #     if not plt.fignum_exists(fignum):
    #         print("user closed plot; exiting")
    #         break
    #     plt.gcf().clear()
    #     # I should probably learn how to write a context manager for this (with plt. .... as plot:) so I don't just keep copy-pasting this code to close the interactive plot
    #     
    #     y = get_fencepost_deviation_sum(x, seed, spectrum_exponent)
    #     xs.append(x)
    #     ys.append(y)
    #     xs = xs[-max_frames:]
    #     ys = ys[-max_frames:]
    #     x += x_step
    #     plt.plot(xs, ys)
    #     plt.draw()
    #     plt.pause(0.01)



if android:
    linspace = lambda a,b,n: [a+i*(b-a)/(n-1) for i in range(n)]
    sign = lambda x: 1 if x > 0 else -1 if x < 0 else 0
else:
    linspace = np.linspace
    sign = np.sign


if __name__ == "__main__":
    # todo: introduce seasonal variation and variation by place
    # todo: joint distribution, events like thunderstorm and tornado are not independent
    # run_simple_simulation()

    # want to get a deterministic function with good autocorrelation like long-term variations, and smaller variations around that trend, etc. to fractal precision

    seed = str(time.time())
    spectrum_exponent = 0.4
    offset = random.uniform(-100000, 100000)
    xs = linspace(0+offset, 1000+offset, 1001)
    ys = [get_fencepost_deviation_sum(x, seed, spectrum_exponent) for x in xs]
    # summarize_xs_ys(xs, ys)
    # print_graph(xs, ys, n_ticks=250)
    # print_graph_live(seed, spectrum_exponent, x0=xs[0], x_step=xs[1]-xs[0])
    plot_live(seed, spectrum_exponent, x0=xs[0], x_step=xs[1]-xs[0])
    if not android:
        plt.plot(xs, ys)
        # plt.show()
        plt.savefig("HashWeatherPlot.png")

    #d = get_test_wave_function_at_fenceposts()
    #xs = sorted(d.keys())
    #ys = [d[x] for x in xs]

    #seed = random.uniform(0, 100)
    #print(f"seed is {seed}")

    #hash_wave_function = get_hash_wave_function(seed_object=seed)

    #xs = np.arange(0, 100, 0.01)
    #ys = hash_wave_function(xs)
    
    if android:
        print("done")
        input()
