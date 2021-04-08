# idea to turn a vector of numbers into a sound wave
# have some list of "basis spectra" which can be activated to different extents and added together

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import Music.WavUtil as wav


MIN_HZ = 440*(2**(-21/12))
MAX_HZ = 440*(2**(+51/12))


class Articulator:
    # methods implemented on subclass-specific basis are omitted here

    def get_spectrum(self):
        f = self.get_spectrum_function()
        l2s = get_log_frequency_domain()
        return f(l2s)

    def get_spectrum_between_positions(self, source_position, target_position, alpha):
        # alpha=0 is completely at source, alpha=1 is completely at target
        # treat the articulator instances as also being positions of the articulator type
        assert 0 <= alpha <= 1
        s0 = source_position.get_spectrum()
        s1 = target_position.get_spectrum()
        return (1-alpha) * s0 + alpha * s1

    def n_params(self):
        return len(self.vector())

    @classmethod
    def from_vector(cls, v):
        return cls(*v)
        

class SpikeArticulator(Articulator):
    def __init__(self, mu, dx):
        self.mu = mu
        self.dx = dx

    def get_spectrum_function(self):
        return spectrum_spike(self.mu, self.dx)

    def vector(self):
        return np.array([self.mu, self.dx])

    @staticmethod
    def in_default_position():
        return SpikeArticulator(mu=np.log2(440), dx=1/12)

    @staticmethod
    def random():
        mu = random.uniform(np.log2(MIN_HZ), np.log2(MAX_HZ))
        dx = abs(np.random.normal(0, 1/2))
        return SpikeArticulator(mu, dx)


class NormalArticulator(Articulator):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def get_spectrum_function(self):
        return spectrum_normal(self.mu, self.sigma)

    def vector(self):
        return np.array([self.mu, self.sigma])

    @staticmethod
    def in_default_position():
        return NormalArticulator(mu=np.log2(440), sigma=1/12)

    @staticmethod
    def random():
        mu = random.uniform(np.log2(MIN_HZ), np.log2(MAX_HZ))
        sigma = abs(np.random.normal(0, 1/2))
        return NormalArticulator(mu, sigma)
    

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
    mus_l2 = np.arange(np.log2(MIN_HZ), np.log2(MAX_HZ), 1/12)
    sigmas_octave_diff = [1/6]
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


def get_spectrum_from_vector_in_basis(vector, average_amplitude=1):
    basis_spectra = get_basis_spectra()
    assert len(vector) == len(basis_spectra)
    l2s = get_log_frequency_domain()
    spectrum = np.zeros((len(l2s),))
    for f, weight in zip(basis_spectra, vector):
        arr_raw = f(l2s)
        arr = weight * arr_raw
        spectrum += arr
    if spectrum.mean() == 0:
        assert average_amplitude == 0, "cannot turn zero-amplitude spectrum into non-zero amplitude"
    else:
        spectrum *= average_amplitude / spectrum.mean()
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


def convert_spectrum_to_waveform(spectrum, rate=44100, seconds=1):
    l2s = get_log_frequency_domain()
    n_samples = round(rate*seconds)
    res = np.zeros((n_samples,))
    xs = np.arange(0, seconds, 1/rate)
    for l2, weight in zip(l2s, spectrum):
        hz = 2**l2
        phase = random.uniform(0, 2*np.pi)  # needed so they don't all add up around x=0
        component = weight * np.sin(hz*2*np.pi*xs + phase)
        res += component
    return res


def convert_spectrum_sequence_to_waveform(spectra, rate=44100, seconds=1):
    n_spectra = len(spectra)
    # no fencepost here, just do a block of static waveform for each spectrum in order
    # if user wants more fine-grained transition, should pass a spectrum sequence reflecting that
    seconds_per_spectrum = seconds/n_spectra
    res = None
    for s in spectra:
        wave = convert_spectrum_to_waveform(s, rate=rate, seconds=seconds_per_spectrum)
        if res is None:
            res = wave
        else:
            res = np.concatenate([res, wave])
    return res


def add_noise_to_spectrum(spectrum, noise_average_amplitude=1):
    noisy_spectrum = get_noisy_spectrum(average_amplitude=noise_average_amplitude)
    return (spectrum + noisy_spectrum)/2


def add_noise_to_spectra(spectra, noise_average_amplitude=1):
    noisy_spectrum = get_noisy_spectrum(average_amplitude=noise_average_amplitude)
    return [(s+noisy_spectrum)/2 for s in spectra]


def get_noisy_spectrum(average_amplitude=1):
    # something like red noise (low frequencies common) with some fluctuation
    l2s = get_log_frequency_domain()
    f = spectrum_spike(mu=np.log2(MIN_HZ), dx=np.log2(MAX_HZ/MIN_HZ))
    factors = np.random.normal(1,0.5,(len(l2s),))
    spectrum = f(l2s) * factors
    spectrum *= average_amplitude / spectrum.mean()
    return spectrum


def get_random_spectrum(average_amplitude=1):
    length = len(get_basis_spectra())
    vector = np.random.uniform(0, 1, (length,))
    return get_spectrum_from_vector_in_basis(vector, average_amplitude=average_amplitude)


def get_zero_spectrum():
    length = len(get_basis_spectra())
    vector = np.zeros((length,))
    return get_spectrum_from_vector_in_basis(vector, average_amplitude=0)


def get_random_spectrum_sequence(n_spectra, frames_per_spectrum, average_amplitude=1):
    z = get_zero_spectrum()
    initial_spectra = [z] + [get_random_spectrum(average_amplitude=average_amplitude) for i in range(n_spectra)] + [z]
    # interpolate linearly between neighboring spectra, also flank by zero-intensity spectra
    # frames_between_spectra = frames_per_spectrum - 1
    res = []
    for s0, s1 in zip(initial_spectra[:-1], initial_spectra[1:]):
        for a in range(frames_per_spectrum):
            if a == 0:
                res.append(s0)
            else:
                alpha = a/frames_per_spectrum
                s = (1-alpha) * s0 + alpha * s1
                res.append(s)

    # now add the last one
    res.append(initial_spectra[-1])
    assert len(res) == 1 + frames_per_spectrum * (len(initial_spectra) - 1)
    return res


def get_articulators():
    return [SpikeArticulator.in_default_position() for i in range(3)] + [NormalArticulator.in_default_position() for i in range(3)]


def get_default_position_vector(articulators):
    res = None
    for art in articulators:
        default_pos = type(art).in_default_position()
        v = default_pos.vector()
        if res is None:
            res = v
        else:
            res = np.concatenate([res, v])
    return res


def get_random_articulation_vector(articulators):
    # this vector encodes information about how each articulator is positioned
    res = None
    for articulator in articulators:
        art = type(articulator).random()
        v = art.vector()
        if res is None:
            res = v
        else:
            res = np.concatenate([res, v])
    return res


def get_random_articulation_vectors(articulators, n_vectors):
    return [get_random_articulation_vector(articulators) for i in range(n_vectors)]


def get_spectrum_from_vector_in_articulation(vector, articulators):
    l2s = get_log_frequency_domain()
    res = np.zeros((len(l2s),))
    for art in articulators:
        n_params = art.n_params()
        sub_vector = vector[:n_params]
        vector = vector[n_params:]
        art = type(art).from_vector(sub_vector)
        spectrum = art.get_spectrum()
        res += spectrum
    return res


def get_spectra_from_vectors_in_articulation(vectors, articulators, frames_per_vector):
    # move alpha through articulator space NOT spectrum space
    # so we want the articulator to have to move its mu value from a to b, for example
    # rather than just a pointwise linear combination of the spectra, which will not get the pitch moving continuously from a to b
    z = get_default_position_vector(articulators)
    initial_vectors = [z] + vectors + [z]
    # interpolate linearly between neighboring articulator positions, also flank by default position vectors (articulatory setting)
    res = []
    for v0, v1 in zip(initial_vectors[:-1], initial_vectors[1:]):
        for a in range(frames_per_vector):
            alpha = a/frames_per_vector
            v = (1-alpha) * v0 + alpha * v1
            spectrum = get_spectrum_from_vector_in_articulation(v, articulators)
            res.append(spectrum)

    # now add the last one
    final_spectrum = get_spectrum_from_vector_in_articulation(initial_vectors[-1], articulators)
    res.append(final_spectrum)
    assert len(res) == 1 + frames_per_vector * (len(initial_vectors) - 1)
    return res


def plot_random_spectrum():
    spectrum = get_random_spectrum()
    plot_spectrum(spectrum)


def plot_spectrum(spectrum, show=True):
    l2s = get_log_frequency_domain()
    plt.plot(l2s, spectrum)
    plt.title("spectrum in log-Hz domain")
    set_xticks()
    if show:
        plt.show()


def plot_spectra(spectra, show=True, equal_ylim=True):
    l2s = get_log_frequency_domain()
    plt.title("spectrum in log-Hz domain")
    max_y = max(s.max() for s in spectra)
    for i, s in enumerate(spectra):
        plt.subplot(len(spectra), 1, i+1)
        plt.plot(l2s, s)
        plt.ylim(0, max_y)
        set_xticks()
    if show:
        plt.show()


def plot_spectrum_and_fft(spectrum, waveform=None):
    if waveform is None:
        waveform = convert_spectrum_to_waveform(spectrum)
    plt.subplot(211)
    plot_spectrum(spectrum, show=False)
    plt.subplot(212)
    fft = np.fft.fft(waveform)
    plt.plot(fft)
    plt.show()



if __name__ == "__main__":
    # plot_basis_spectra()
    # spectrum = get_random_spectrum(average_amplitude=1)
    # spectra = get_random_spectrum_sequence(n_spectra=7, frames_per_spectrum=5, average_amplitude=1)
    # noisy_spectra = add_noise_to_spectra(spectra, noise_average_amplitude=0.5)
    # wav.write_signal_to_wav(waveform, "BasisSpectraOutput.wav")
    # plot_spectrum_and_fft(spectrum, waveform)

    # convert to and from sound seems too complicated/slow for now, just do spectrum with some distortion as the input to the other agents
    # noisy_spectrum = add_noise_to_spectrum(spectrum, noise_average_amplitude=0.5)
    # noisy_waveform = convert_spectrum_to_waveform(noisy_spectrum, seconds=1)
    # signal = np.concatenate([waveform, noisy_waveform])
    # plot_spectra([spectrum, noisy_spectrum])

    articulators = get_articulators()
    # articulation_vector = get_random_articulation_vector(articulators)
    articulation_vectors = get_random_articulation_vectors(articulators, n_vectors=4)
    print(articulation_vectors)
    # spectrum = get_spectrum_from_vector_in_articulation(articulation_vector, articulators)
    spectra = get_spectra_from_vectors_in_articulation(articulation_vectors, articulators, frames_per_vector=20)
    # plot_spectrum(spectrum)
    # plot_spectra(spectra)
    # signal = convert_spectrum_to_waveform(spectrum, seconds=1)
    signal = convert_spectrum_sequence_to_waveform(spectra, seconds=3)

    # signal = convert_spectrum_sequence_to_waveform(noisy_spectra, seconds=3)
    wav.write_signal_to_wav(signal, "BasisSpectraOutput.wav")
    # plot_spectra(spectra)
