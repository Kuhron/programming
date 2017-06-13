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


def get_similarity_between_spectra(spec1, spec2):
    return anderson_ksamp([spec1, spec2]).statistic  # who needs significance level


def convert_wav_to_spectrogram(filepath):
    array = wav.read_wav_to_array(filepath)
    spec = plt.specgram(array, Fs=wav.RATE)
    plt.gca().set_ylim((0, 6000))
    return spec
    # return spectrogram(array, fs=wav.RATE)


def compare_spectrograms(spectrogram1, spectrogram2):
    shorter, longer = sorted([spectrogram1, spectrogram2], key=len)
    r = len(longer) *1.0/ len(shorter)
    similarities = []
    for i in range(len(shorter)):
        j = int(r * i)
        similarity = get_similarity_between_spectra(shorter[i], longer[j])
        similarities.append(similarity)
    return np.mean(similarities)


def compare_wavs(fp1, fp2):
    Pxx1, freqs1, bins1, im1 = convert_wav_to_spectrogram(fp1)
    Pxx2, freqs2, bins2, im2 = convert_wav_to_spectrogram(fp2)
    res = compare_spectrograms(Pxx1, Pxx2)
    # assert res == compare_spectrograms(Pxx2, Pxx1)
    return res


if __name__ == "__main__":
    fps = [
        "test.wav",
        "test2.wav",
        "test3.wav",
        "test4.wav",
        "test5.wav",
    ]
    # files = random.sample(fps, 2)  # OverflowError with certain files?
    files = fps[:2]
    print(files)
    similarity = compare_wavs(*files)
    print(similarity)