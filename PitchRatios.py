import math, winsound
import numpy as np

min_freq = 37
max_freq = 32767

base_octave_pitches = {}
base_octave_notes = ["a","a#","b","c","c#","d","d#","e","f","f#","g","g#", "a\'"]
for i in range(13):
    base_octave_pitches[base_octave_notes[i]] = 220.0*((2.0**(1.0/12.0))**i)
#print(base_octave_pitches)

def get_frequency_from_note_name(s, shift=0):
    if s is None:
        return None

    s = s.upper()

    if any([i not in "ABCDEFG#b0123456789" for i in s]) or len(s) > 3:
        raise ValueError("Invalid note name {0}".format(s))
    if s[-1] not in "0123456789":
        s += "4"

    n = s[:-1]
    o = s[-1]
    v = "A_BC_D_EF_G_".index(n[0])
    if len(n) > 1:
        v += (1 if n[1] == "#" else -1 if n[1] == "b" else ValueError)

    o = int(o) - 4
    v += 12 * o + shift
    # A4 = 440 Hz is v = 0
    return int(440 * 2**(v * 1.0/12))

def get_initial_pitch():
    p = input("What is the initial pitch? If note name, enter lowercase (use # for accidental). Else, enter hertz.\n")
    # if p in base_octave_pitches:
    #     return base_octave_pitches[p] # all values are floats
    # else:
    #     return float(p)
    return get_frequency_from_note_name(p, 0)

def get_log_interval():
    frac = input("What proportion of an octave should the step size be?\n")
    if "/" in frac:
        numerator,denominator = frac.split("/")
        return float(numerator)/float(denominator)
    else:
        return float(frac)

def get_ratio_series_params():
    s = input("Enter the minimum ratio, maximum ratio, and step size, separated by spaces.\n")
    s = s.split()[:3]
    return [float(i) for i in s]

def log_interval_to_ratio(log_interval):
    return 2.0**log_interval

def get_ratio():
    return log_interval_to_ratio(get_log_interval())

def num_notes(initial_pitch, ratio):
    if initial_pitch < min_freq or initial_pitch > max_freq:
        return 0
    # make it tell the maximum number of notes that can be played given the initial and the ratio
    return -1

def play_octave():
    #fluidsynth.play_Note("C-5")

    p = get_initial_pitch()
    v = get_log_interval()
    r = log_interval_to_ratio(v)
    freqs = []
    for i in range(math.floor(2/v)+1):
        f = p*(r**i)
        freqs.append(int(f))
    for i in freqs:
        winsound.Beep(i, 500)

def play_ratio_series():
    p = get_initial_pitch()
    r_min, r_max, step = get_ratio_series_params()
    freqs = []
    for r in np.arange(r_min, r_max + step, step):
        f = p*r
        freqs.append(int(f))
    for i in freqs:
        print("%d Hz" % i)
        winsound.Beep(i, 800)

def play_ratio_set():
    p = get_initial_pitch()
    rs = [float(i) for i in input("ratios, separated by spaces: ").split()]
    freqs = []
    for r in rs:
        f = p*r
        freqs.append(int(f))
    for i in freqs:
        print("%d Hz" % i)
        winsound.Beep(i, 800)



action = input("1. play octave with n equal steps\n"\
    "2. play series of pitches by ratio over base pitch\n"\
    "3. play custom series of pitch ratios over base pitch\n")

if action == "1":
    play_octave()
elif action == "2":
    play_ratio_series()
elif action == "3":
    play_ratio_set()
else:
    print("invalid option")


















