import matplotlib.pyplot as plt
import numpy as np


def get_amp_from_freq(freq):
    # determine noise type; this function simply gives the fourier transform's value at a given frequency
    return freq
    # return np.log(1 + freq)

def get_wave(xs, freq):
    amp = get_amp_from_freq(freq)
    phase = np.random.uniform(0, 2 * np.pi)
    return amp * np.sin(xs / freq + phase)


def get_waves_from_sequence(xs, seq):
    # use generator to make lists in constant space rather than creating them all first before processing
    for freq in seq:
        yield get_wave(xs, freq)


def get_wave_sum(xs, max_period, period_step):
    freqs = 1 / np.arange(period_step, max_period, period_step)
    waves = get_waves_from_sequence(xs, freqs)
    res = np.array([0.0] * len(xs))
    for wave in waves:
        res += wave
    return res


if __name__ == "__main__":
    xs = np.arange(0, 100, 0.001)
    wave = get_wave_sum(xs, 10, 0.001)
    plt.plot(xs, wave)
    plt.show()