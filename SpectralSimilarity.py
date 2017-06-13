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
    return anderson_ksamp([spec1, spec2])


def convert_wav_to_spectrogram(filepath):
    array = wav.read_wav_to_array(filepath)
    spec = plt.specgram(array, Fs=wav.RATE)
    plt.gca().set_ylim((0, 6000))
    return spec
    # return spectrogram(array, fs=wav.RATE)


def compare_spectrograms(spec1, spec2):
    raise


if __name__ == "__main__":
    fp1 = "test.wav"
    Pxx1, freqs1, bins1, im1 = convert_wav_to_spectrogram(fp1)
    print(Pxx1.shape)  # (len(freqs), len(bins))
    # plt.show()