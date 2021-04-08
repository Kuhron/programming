# idea to turn a vector of numbers into a sound wave
# have some list of "basis spectra" which can be activated to different extents and added together

import numpy as np
import matplotlib.pyplot as plt
import math


MIN_HZ = 100
MAX_HZ = 10000


def spectrum_normal(mu, sigma):
    # beware lambda closure
    return lambda x, mu=mu, sigma=sigma: 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-mu)/sigma)**2)


def spectrum_spike(mu, dx):
    # a triangle with peak at mu, extending to dx on each side
    return lambda x, b=mu, a=dx: 1/a * np.maximum(0, a-abs(x-b))


def get_log_frequency_domain():
    min_l2 = np.log2(MIN_HZ)
    max_l2 = np.log2(MAX_HZ)
    l2s = np.linspace(min_l2, max_l2, 1000)
    return l2s


def get_basis_spectra():
    l2s = get_log_frequency_domain()
    mus_l2 = np.linspace(np.log2(MIN_HZ), np.log2(MAX_HZ), 30)
    sigmas_octave_diff = [1]
    # functions = [spectrum_normal(mu=mu, sigma=octave_diff) for mu in mus_l2 for octave_diff in sigmas_octave_diff]
    functions = [spectrum_spike(mu=mu, dx=octave_diff) for mu in mus_l2 for octave_diff in sigmas_octave_diff]
    return functions


def get_xtick_values_hz():
    # count within each order of mag like a nice log graph scale e.g. 100 200 ... 900 1000 2000 ... 9000 10000 20000 ...
    min_hz_order_mag = math.floor(np.log10(MIN_HZ))
    max_hz_order_mag = math.floor(np.log10(MAX_HZ))
    res = []
    current = MIN_HZ
    current_order_mag = min_hz_order_mag
    while True:
        res.append(current)
        if current >= MAX_HZ:
            break
        current_power_10 = 10 ** current_order_mag
        current += current_power_10
        if current // current_power_10 == 10:
            # go ahead and increment the power now, this newly-made current will get added next iteration and then the bigger power will get added
            current_order_mag += 1
    return res


def is_power_of_10(x):
    return np.log10(x) % 1 == 0


def get_xtick_labels(xtick_values_hz):
    return [str(x) if is_power_of_10(x) else "" for x in xtick_values_hz]


def set_xticks():
    xticks_hz = get_xtick_values_hz()
    xticks_log2 = np.log2(xticks_hz)
    xtick_labels = get_xtick_labels(xticks_hz)
    # print(xtick_labels)
    plt.xticks(xticks_log2)
    plt.gca().set_xticklabels(xtick_labels)


def plot_basis_spectra():
    basis_spectra = get_basis_spectra()
    l2s = get_log_frequency_domain()
    hzs = 2**l2s
    for f in basis_spectra:
        ys = f(l2s)
        plt.plot(l2s, ys)
    set_xticks()
    plt.show()


def get_spectrum(vector):
    basis_spectra = get_basis_spectra()
    assert len(vector) == len(basis_spectra)
    l2s = get_log_frequency_domain()
    spectrum = np.zeros((len(l2s),))
    for f, weight in zip(basis_spectra, vector):
        arr_raw = f(l2s)
        arr = weight * arr_raw
        spectrum += arr
    return spectrum


def convert_spectrum_to_hz_domain(spectrum):
    res = [0 for hz in range(1, MIN_HZ)]
    l2_range = np.log2(MAX_HZ/MIN_HZ)  # subtraction/division property of logs
    min_l2 = np.log2(MIN_HZ)
    for hz in range(MIN_HZ, MAX_HZ+1):
        l2 = np.log2(hz)
        index_in_spectrum_array = int((l2-min_l2)/l2_range)
        res.append(spectrum[index_in_spectrum_array])
    return res


def get_random_spectrum():
    length = len(get_basis_spectra())
    vector = np.random.uniform(0, 1, (length,))
    return get_spectrum(vector)


def plot_random_spectrum():
    spectrum = get_random_spectrum()
    plot_spectrum(spectrum)


def plot_spectrum(spectrum):
    l2s = get_log_frequency_domain()
    plt.plot(l2s, spectrum)
    plt.title("spectrum in log-Hz domain")
    set_xticks()
    plt.show()




if __name__ == "__main__":
    plot_basis_spectra()
    spectrum = get_random_spectrum()
    spectrum_hz = convert_spectrum_to_hz_domain(spectrum)
    ifft = np.fft.ifft(spectrum_hz, n=100000)
    plt.plot(ifft)
    plt.show()
    plot_spectrum(spectrum)
    # still failing to get the right sound wave from spectrum
