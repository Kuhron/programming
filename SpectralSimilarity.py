import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import anderson_ksamp
from scipy.signal import spectrogram

import Music.WavUtil as wav


# Anderson-Darling test integrates square of difference between two distribution functions (empirical or parametric)
# the variable of integration is F(x), the cumulative distribution value, of one of the distributions
# visualize a QQ plot, with one EDF just being the y=x line (this is the variable of integration)
# the statistic is the integral of the square of the difference between the y=x line and the other distribution on the plot


def get_distance_between_samples(s1, s2):
    return anderson_ksamp([s1, s2]).statistic  # who needs significance level


def get_sample_from_spectrum(spec, freqs, n_observations):
    # treat spectrum as pdf over the frequencies
    norm_spec = spec / np.sum(spec)
    return np.random.choice(freqs, p=norm_spec, size=(n_observations,))


def convert_wav_to_spectrogram(filepath):
    array = wav.read_wav_to_array(filepath)
    spec = plt.specgram(array, Fs=wav.RATE)
    plt.gca().set_ylim((0, 6000))
    return spec
    # return spectrogram(array, fs=wav.RATE)


def compare_spectrograms(spectrogram1, spectrogram2, freqs1, freqs2):
    shorter, longer = sorted([spectrogram1, spectrogram2], key=len)
    r = len(longer) *1.0/ len(shorter)
    distances = []
    n_observations = 100  # make this too high and it will overflow; 100 is good enough since a wav will have <1 distance to itself
    for i in range(len(shorter)):
        # line up the time axes by just dilating (no translating or dilating by a changing factor yet)
        # bootstrap some frequencies from each signal with probability=intensity
        # compare these two samples' EDFs
        j = int(r * i)
        pdf1 = shorter[:, i]
        pdf2 = longer[:, j]
        samp1 = get_sample_from_spectrum(pdf1, freqs1, n_observations)
        samp2 = get_sample_from_spectrum(pdf2, freqs2, n_observations)
        distance = get_distance_between_samples(samp1, samp2)
        distances.append(distance)
    return np.mean(distances)


def compare_wavs(fp1, fp2):
    Pxx1, freqs1, bins1, im1 = convert_wav_to_spectrogram(fp1)
    Pxx2, freqs2, bins2, im2 = convert_wav_to_spectrogram(fp2)
    res = compare_spectrograms(Pxx1, Pxx2, freqs1, freqs2)
    # assert res == compare_spectrograms(Pxx2, Pxx1)
    return res


def get_distance_matrix(fps):
    n = len(fps)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            distance = compare_wavs(fps[i], fps[j])
            res[i][j] = res[j][i] = distance
    return res


if __name__ == "__main__":
    fps = [
        "test.wav",   # testing 1 2 3 a e i o u
        "test2.wav",  # [same as test.wav, but just said again; should be very similar]
        "test3.wav",  # [hitting microphone with fingernails]
        "test4.wav",  # asdfasdfblabla different vowels not what I said before [should be much more similar to 1 and 2 than to 3 and 5]
        "test5.wav",  # [blowing on microphone]
    ]
    # files = random.sample(fps, 2)  # OverflowError with certain files?
    # files = fps[:2]  # should work
    # print(files)
    # similarity = compare_wavs(*files)
    # print(similarity)
    np.set_printoptions(suppress=True)
    print(get_distance_matrix(fps))

    # expected result:
    # [[   0   low   high  lowish  high]
    #  [         0   high  lowish  high]
    #  [                0    high  high]
    #  [                        0  high]
    #  [                              0]]

    # problems:
    # - only dilates time axis by constant factor rather than "squishing" the durations of similar spectra (as with multiple people saying the same word)
    # - sees different frequencies as dissimilar, so would reject speakers with different pitch of voice as dissimilar no matter what