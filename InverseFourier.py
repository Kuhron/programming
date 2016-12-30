import math
import random
import wave

import numpy as np
import matplotlib.pyplot as plt

from FunctionalFormGui import get_function, get_distribution_function
import Music.WavUtil as wav


def elementwise_mean(array):
    return np.mean(array, axis=0)

def function_sum(fs):
    return lambda x: sum(f(x) for f in fs)

def function_mean(fs):
    return lambda x: function_sum(fs)(x) / len(fs)

def get_test_signal():
    # http://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file
    with wave.open("test_wav.wav", "r") as spf:
        assert spf.getnchannels() == 1, "Please use mono wav file"
        wav_params = spf.getparams()
        signal = spf.readframes(-1)

    signal = np.fromstring(signal, "Int16")
    # plt.plot(signal)
    # plt.show()


def write_wav_from_freqs(freqs, n_seconds):
    fps = wav.RATE
    n_frames = fps * n_seconds

    signal_function_from_freq = lambda freq: lambda i: np.sin(i / fps * freq * (2*np.pi) + np.random.uniform(0, 2*np.pi))

    signal_functions = [signal_function_from_freq(freq) for freq in freqs]
    new_signal_function = function_mean(signal_functions)
    new_signal = new_signal_function(np.array([i for i in range(n_frames)]))

    wav.write_signal_to_wav(new_signal, "InverseFourierOutput.wav")


def test_can_open_file():
    try:
        with wave.open("InverseFourierOutput.wav", "w") as spf:
            spf.setnchannels(1)
            spf.setsampwidth(2)
            spf.setframerate(44100)
            spf.setnframes(0)
    except:
        raise RuntimeError("close the file!")


def get_freqs(n_freqs, min_freq, max_freq, restrict_to_notes=False):
    test_can_open_file()
    f = get_function(min_freq, max_freq, 0, 1)
    xs = range(min_freq, max_freq + 1)
    ys = [f(x) for x in xs]

    dist = get_distribution_function(xs, f)
    freqs = [dist() for i in range(n_freqs)]
    freqs = [max(1, x) for x in freqs]

    if restrict_to_notes:
        NOTES_PER_OCTAVE = 12
        OCTAVE_FACTOR = 2
        log_freqs = [math.log(freq, OCTAVE_FACTOR) for freq in freqs]
        rounded_logs = [round(x * NOTES_PER_OCTAVE, 0) / NOTES_PER_OCTAVE for x in log_freqs]
        offset_440 = math.log(440, OCTAVE_FACTOR) % (1 / NOTES_PER_OCTAVE)
        freqs = [OCTAVE_FACTOR ** (x + offset_440) for x in rounded_logs]

    return freqs


def get_constant_spectrum(freqs, n_seconds):
    write_wav_from_freqs(freqs, n_seconds)

    plt.hist(freqs)
    plt.savefig("InverseFourierDist.png")
    plt.show()


def get_varying_spectrum(freqs, full_length_seconds):
    fps = wav.RATE
    n_frames = fps * n_seconds

    peaks = [random.uniform(0, full_length_seconds * fps) for _ in freqs]
    get_length_dilation = lambda: random.uniform(2, 10)

    def amplitude_function(peak):
        dilation = get_length_dilation()
        return lambda t: 1 / (1 + ((t - peak) / dilation / fps) ** 4)  # greater than 1/2 from -1 to 1, so dilate this interval

    phase_function = lambda freq: lambda t: np.sin(t / fps * freq * (2*np.pi) + np.random.uniform(0, 2*np.pi))
    amplitudes = []
    phases = []

    def get_signal_function(freq, peak):
        amplitude = amplitude_function(peak)
        amplitudes.append(amplitude)
        phase = phase_function(freq)
        phases.append(phase)
        return lambda t: amplitude(t) * phase(t)

    signal_functions = [get_signal_function(freq, peak) for peak, freq in zip(peaks, freqs)]
    new_signal_function = function_mean(signal_functions)
    xs = np.array([i for i in range(n_frames)])
    new_signal = new_signal_function(xs)

    wav.write_signal_to_wav(new_signal, "InverseFourierOutput.wav")

    # trying to create spectrogram; matplotlib's specgram doesn't work well at all for this
    bin_width = 1 / 12
    log_freqs = [math.log(freq, 2) for freq in freqs]
    freq_bins = np.arange(min(log_freqs) - 1, max(log_freqs) + 1, bin_width)
    freq_and_amplitude = [(f, a) for f, p, a in sorted(zip(log_freqs, peaks, amplitudes))]
    rough_xs = np.array([xs[i * 1000] for i in range(math.ceil(len(xs) / 1000))])
    amplitude_array = []
    for lower_bound, upper_bound in zip(freq_bins[:-1], freq_bins[1:]):
        if len(freq_and_amplitude) == 0:
            amplitude_array.append(rough_xs * 0)
            continue
        current_amplitude = rough_xs * 0
        in_bin = lambda freq_and_amplitude, lower_bound, upper_bound: lower_bound <= freq_and_amplitude[0][0] < upper_bound
        while len(freq_and_amplitude) > 0 and in_bin(freq_and_amplitude, lower_bound, upper_bound):
            amplitude = freq_and_amplitude[0][1]
            new_amplitude = amplitude(rough_xs) * 1e6  # plot fails to pick up on tiny values
            current_amplitude += new_amplitude
            freq_and_amplitude = freq_and_amplitude[1:]
        amplitude_array.append(current_amplitude)
    amplitude_array = np.array(amplitude_array)
    aspect_ratio = 1
    aspect_arg = aspect_ratio*(amplitude_array.shape[1] / amplitude_array.shape[0])  # http://stackoverflow.com/questions/11776663/
    plt.imshow(amplitude_array, aspect=aspect_arg, interpolation="none")
    plt.savefig("InverseFourierSpectrum.png")
    # plt.show()


if __name__ == "__main__":
    n_freqs = 100
    n_seconds = 60
    min_freq = 0
    max_freq = 2000
    restrict_to_notes = True

    freqs = get_freqs(n_freqs, min_freq, max_freq, restrict_to_notes)

    # get_constant_spectrum(freqs, n_seconds)
    get_varying_spectrum(freqs, n_seconds)

