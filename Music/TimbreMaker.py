import random
import math
import Music.WavUtil as w
import matplotlib.pyplot as plt
import numpy as np


def get_spectrum_random(n_harmonics):
    return [random.random() for h in range(n_harmonics)]


def get_spectrum_exponential(n_harmonics, random_modification=False):
    r = random.random()
    assert 0 < r < 1
    res = []
    intensity = 1
    for h in range(n_harmonics):
        res.append(intensity)  # first one should be 1
        intensity *= r
    return res


def modify_spectrum_random(spectrum):
    res = []
    sigma = random.uniform(0, 1)
    for intensity in spectrum:
        x = random.normalvariate(0, sigma)
        multiplier = math.exp(x)
        new_intensity = intensity * multiplier
        res.append(new_intensity)
    return res


if __name__ == "__main__":
    f0 = 200
    n_harmonics = 12
    harmonics = [f0*i for i in range(1, n_harmonics+1)]
    # harmonics = [f0*x for x in np.arange(1, n_harmonics+1, 0.01)]  # for making noise

    # harmonic_weights = get_spectrum_random(len(harmonics))
    harmonic_weights = get_spectrum_exponential(len(harmonics), random_modification=True)

    seconds = 2
    unweighted_signals = [w.get_signal_from_freq(f, seconds, truncate=False) for f in harmonics]
    weighted_signals = [s*weight for s, weight in zip(unweighted_signals, harmonic_weights)]
    total_signal = sum(weighted_signals)
    for f, weight in zip(harmonics, harmonic_weights):
        print("{} Hz * {}".format(f, weight))
    w.send_signal_to_audio_out(total_signal)
    plt.scatter(harmonics, harmonic_weights)
    plt.show()
