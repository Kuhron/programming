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


def note_number_to_hertz(n):
    # a440 is 69, c0 ~=16 Hz is 0
    deviation_from_a440 = (n - 69)/12
    a440_freq = 440
    factor_deviation = 2 ** deviation_from_a440
    return a440_freq * factor_deviation


def hertz_to_note_number(hz):
    # a440 is 69, c0 ~=16 Hz is 0
    log2_hz = math.log(hz, 2)
    log2_a440 = math.log(440, 2)
    deviation_in_logs = log2_hz - log2_a440
    deviation_in_semitones = deviation_in_logs*12
    return 69 + deviation_in_semitones


def note_number_to_name(n):
    pitch_class = n % 12
    octave = (n // 12) - 1  # C-1 ~= 8 Hz is number 0; C0 ~= 16 Hz is number 12
    if pitch_class % 1 == 0:
        pitch_class = int(pitch_class)
        letter = "CKDHEFXGJARB"[pitch_class]
    else:
        letter = "?"
    assert octave % 1 == 0
    octave = int(octave)
    return letter + str(octave)


def test_note_number_math():
    assert hertz_to_note_number(440) == 69, "440 Hz is #{}, not #69".format(hertz_to_note_number(440))

    n = 0
    h = note_number_to_hertz(n)
    nn = hertz_to_note_number(h)
    assert abs(nn - n) < 1e-6, "n={}, h={}, nn={}".format(n, h, nn)

    n = 69 + 12
    h = note_number_to_hertz(n)
    assert abs(h - 880) < 1e-6, "n={}, h={}, should be 880".format(n, h)
    nn = hertz_to_note_number(880)
    assert abs(nn - n) < 1e-6, "nn from 880 = {}, should be {}".format(nn, n)

    n = -25
    h = note_number_to_hertz(n)
    nn = hertz_to_note_number(h)
    assert abs(nn - n) < 1e-6, "n={}, h={}, nn={}".format(n, h, nn)

    n = 106.1315126
    h = note_number_to_hertz(n)
    nn = hertz_to_note_number(h)
    assert abs(nn - n) < 1e-6, "n={}, h={}, nn={}".format(n, h, nn)

    n = np.pi * np.exp(np.pi)
    h = note_number_to_hertz(n)
    nn = hertz_to_note_number(h)
    assert abs(nn - n) < 1e-6, "n={}, h={}, nn={}".format(n, h, nn)


def get_freqs(n_freqs, min_note_number, max_note_number, restrict_to_notes=True):
    NOTES_PER_OCTAVE = 12
    OCTAVE_FACTOR = 2

    test_can_open_file()
    # min_note_number = hertz_to_note_number(min_freq)
    # max_note_number = hertz_to_note_number(max_freq)
    if restrict_to_notes:
        min_note_number = math.floor(min_note_number)
        max_note_number = math.floor(max_note_number)

    f = get_function(min_note_number, max_note_number, 0, 1)
    xs = np.arange(min_note_number, max_note_number + 1, 0.01)  # max precision needed is cents
    ys = [f(x) for x in xs]
    # print("distribution values:")
    # for x, y in zip(xs, ys):
    #     print("{} : {}".format(x, y))

    dist = get_distribution_function(xs, f)
    dist_outputs = sorted([dist() for i in range(n_freqs)])

    if restrict_to_notes:
        # round all the note numbers in the dist_outputs
        rounded_dist_outputs = [round(x, 0) for x in dist_outputs]
        freqs = [note_number_to_hertz(n) for n in rounded_dist_outputs]
        freq_names = [note_number_to_name(n) for n in rounded_dist_outputs]
    else:
        freqs = [note_number_to_hertz(n) for n in dist_outputs]
        freq_names = None

    return freqs, freq_names


def get_constant_spectrum(freqs, n_seconds):
    write_wav_from_freqs(freqs, n_seconds)

    plt.hist(freqs)
    plt.savefig("InverseFourierDist.png")
    plt.show()


def get_varying_spectrum(freqs, freq_names, full_length_seconds, length_dilation_dist):
    fps = wav.RATE
    n_frames = fps * full_length_seconds

    peaks = [random.uniform(0, full_length_seconds * fps) for _ in freqs]

    get_length_dilation = length_dilation_dist
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

    signal_functions = []
    signal_i = 0
    for peak, freq in zip(peaks, freqs):
        if signal_i % 10 == 0:
            print("getting signal #{}/{}".format(signal_i, len(freqs)))
        func = get_signal_function(freq, peak)
        signal_functions.append(func)

    new_signal_function = function_mean(signal_functions)
    xs = np.array([i for i in range(n_frames)])
    new_signal = new_signal_function(xs)

    wav.write_signal_to_wav(new_signal, "InverseFourierOutput.wav")

    # trying to create spectrogram; matplotlib's specgram doesn't work well at all for this
    bin_width = 1 / 12
    log_freqs = [math.log(freq, 2) for freq in freqs]
    freq_bins = np.arange(min(log_freqs) - 1, max(log_freqs) + 1, bin_width)
    freq_and_amplitude = [(f, a) for f, p, a in sorted(zip(log_freqs, peaks, amplitudes))]

    graph_x_points_per_second = 50
    x_step = int(fps/graph_x_points_per_second)  # e.g. fps=44100, graph_x_points_per_second=50 --> x_step = 882
    print("plotting every {} frames = {} times per second".format(x_step, graph_x_points_per_second))
    graph_x_points = np.array([xs[i * x_step] for i in range(math.ceil(len(xs) / x_step))])  # only have some x points to avoid massive image
    # want xticks each second
    x_tick_labels = [i for i in range(full_length_seconds + 1)]  # actual time-unit labels on axis
    x_ticks = [i * graph_x_points_per_second for i in x_tick_labels]
    xlabel = "seconds"

    

    amplitude_array = []
    for lower_bound, upper_bound in zip(freq_bins[:-1], freq_bins[1:]):
        if len(freq_and_amplitude) == 0:
            amplitude_array.append(graph_x_points * 0.0)
            continue
        current_amplitude = graph_x_points * 0.0  # need float because will get complaint at "current_amplitude += new_amplitude"
        in_bin = lambda freq_and_amplitude, lower_bound, upper_bound: lower_bound <= freq_and_amplitude[0][0] < upper_bound
        while len(freq_and_amplitude) > 0 and in_bin(freq_and_amplitude, lower_bound, upper_bound):
            amplitude = freq_and_amplitude[0][1]
            new_amplitude = amplitude(graph_x_points) * 1e6  # plot fails to pick up on tiny values
            current_amplitude += new_amplitude
            freq_and_amplitude = freq_and_amplitude[1:]
        amplitude_array.append(current_amplitude)
    amplitude_array = np.array(amplitude_array)
    aspect_ratio = 1
    aspect_arg = aspect_ratio*(amplitude_array.shape[1] / amplitude_array.shape[0])  # http://stackoverflow.com/questions/11776663/
    plt.imshow(amplitude_array, aspect=aspect_arg, interpolation="none", origin="lower")
    plt.xticks(ticks=x_ticks, labels=x_tick_labels)
    plt.xlabel(xlabel)
    plt.savefig("InverseFourierSpectrum.png")
    # plt.show()


if __name__ == "__main__":
    test_note_number_math()

    n_freqs = 1000
    n_seconds = 60
    length_dilation_dist = lambda: abs(np.random.normal(3, 1))
    min_note_number = 0
    max_note_number = 88

    freqs, freq_names = get_freqs(n_freqs, min_note_number, max_note_number)
    if freq_names is not None:
        for freq, name in zip(freqs, freq_names):
            print("{} = {} Hz".format(name, freq))

    # get_constant_spectrum(freqs, n_seconds)
    get_varying_spectrum(freqs, freq_names, n_seconds, length_dilation_dist)

