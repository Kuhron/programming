# idea to turn a vector of numbers into a sound wave
# have some list of "basis spectra" which can be activated to different extents and added together

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import Music.WavUtil as wav


MIN_HZ = 440*(2**(-4 + 3/12))
MAX_HZ = 440*(2**(+4 + 3/12))
MIN_L2HZ = np.log2(MIN_HZ)
MAX_L2HZ = np.log2(MAX_HZ)

# desmos calculator about potential way to get a formant envelope over harmonics: https://www.desmos.com/calculator/ujriaeh4ps
# f1(x), f2(x), and f3(x) are normal distributions with mus m1 m2 m3 and sigmas s1 s2 s3, with all of those params restricted to interval [0,1] and x also in [0,1] (supposed to represent the log2-hz domain from min_hz to max_hz)
# importantly, I removed the 1/sigma*sqrt(2*pi) for the normal distributions so the mode peak would always be at a height of 1: f1(x) = exp(-1/2 * ((x-m1)/s1)^2), simil f2 and f3
# g123(x) is the average of f1 f2 f3
# could easily add more formants by adding more fs and putting them in the g average
# the actual function showing the spectrum is h123(x) = h(x) * g123(x)
# h(x) is the bump function at harmonics, has a parameter t which is the "tolerance", i.e. the amount of deviation from a harmonic where the h(x) function should have support; it can range from 0 (no support anywhere) to 0.5 (support everywhere); I was using about 0.1 and it looked good
# h(x) = j(x) where 1/2 - abs(1/F0 * mod(2^(Fmin+x*Fmax),F0) - 1/2 <= t; 0 elsewhere
# j(x) is the cosine wave creating the bumps at the right places in h(x)
# j(x) = cos^2 (pi/(2*t) * (1/2 - abs(1/F0 * mod(2^(Fmin+x*Fmax),F0) - 1/2)))
# F0 is a param, the fundamental frequency
# Fmin and Fmax are supposed to help x be a scaling parameter in log2-hz domain (by having the term Fmin + x*Fmax which should give a log2-freq with x being an alpha along that scale), but I don't think this quite works right yet (as of 2021-04-11)


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

    def extra_init_kwargs(self):
        return {}

    @classmethod
    def from_vector(cls, v, **kwargs):
        return cls(*v, **kwargs)
        

class SpikeArticulator(Articulator):
    MIN_MU = np.log2(MIN_HZ)
    MAX_MU = np.log2(MAX_HZ)
    MIN_DX = 1/1200
    MAX_DX = 1/6

    def __init__(self, mu_01, dx_01):
        assert 0 <= mu_01 <= 1, mu_01
        self.mu_01 = mu_01
        assert 0 <= dx_01 <= 1, dx_01
        self.dx_01 = dx_01
        self.mu = SpikeArticulator.MIN_MU + mu_01 * (SpikeArticulator.MAX_MU - SpikeArticulator.MIN_MU)
        self.dx = SpikeArticulator.MIN_DX + dx_01 * (SpikeArticulator.MAX_DX - SpikeArticulator.MIN_DX)

    def get_spectrum_function(self):
        return spectrum_spike(self.mu, self.dx)

    def vector(self):
        return np.array([self.mu_01, self.dx_01])

    @staticmethod
    def in_default_position():
        return SpikeArticulator(mu_01=0.5, dx_01=0.5)

    @staticmethod
    def random():
        mu_01 = random.random()
        dx_01 = random.random()
        return SpikeArticulator(mu_01, dx_01)


class NormalArticulator(Articulator):
    MIN_MU = np.log2(MIN_HZ)
    MAX_MU = np.log2(MAX_HZ)
    MIN_SIGMA = 1/1200
    MAX_SIGMA = 1/12

    def __init__(self, mu_01, sigma_01):
        assert 0 <= mu_01 <= 1, mu_01
        self.mu_01 = mu_01
        assert 0 <= sigma_01 <= 1, sigma_01
        self.sigma_01 = sigma_01
        self.mu = NormalArticulator.MIN_MU + mu_01 * (NormalArticulator.MAX_MU - NormalArticulator.MIN_MU)
        self.sigma = NormalArticulator.MIN_SIGMA + sigma_01 * (NormalArticulator.MAX_SIGMA - NormalArticulator.MIN_SIGMA)
   
    def get_spectrum_function(self):
        return spectrum_normal(self.mu, self.sigma)

    def vector(self):
        return np.array([self.mu_01, self.sigma_01])

    @staticmethod
    def in_default_position():
        return NormalArticulator(mu_01=0.5, sigma_01=0.5)

    @staticmethod
    def random():
        mu_01 = random.random()
        sigma_01 = random.random()
        return NormalArticulator(mu_01, sigma_01)


class ExpCosSquaredArticulator(Articulator):
    def __init__(self, omega_01, phase_01, omega_min, omega_max):
        self.omega_01 = omega_01
        self.phase_01 = phase_01
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.omega = omega_min + self.omega_01 * (omega_max - omega_min)
        self.phase = self.phase_01 * np.pi  # period of cos^2 is pi, so if it goes 0 to pi then 01 of 0 will give similar result as 01 of 1

    def get_spectrum_function(self):
        return spectrum_exp_cos_squared(self.omega, self.phase)

    def vector(self):
        # DON'T train the omega_min and omega_max
        return np.array([self.omega_01, self.phase_01])

    def extra_init_kwargs(self):
        return {"omega_min": self.omega_min, "omega_max": self.omega_max}

    @staticmethod
    def in_default_position(omega_min, omega_max):
        return ExpCosSquaredArticulator(omega_01=0.5, phase_01=0.5, omega_min=omega_min, omega_max=omega_max)

    @staticmethod
    def random():
        omega_01 = random.random()
        phase_01 = random.random()
        return ExpCosSquaredArticulator(omega_01, phase_01, omega_min=5/2*np.pi, omega_max=31/2*np.pi)
 

def spectrum_normal(mu, sigma):
    # beware lambda closure
    return lambda x, mu=mu, sigma=sigma: 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-mu)/sigma)**2)


def spectrum_spike(mu, dx):
    # a triangle with peak at mu, extending to dx on each side
    return lambda x, b=mu, a=dx: 1/a * np.maximum(0, a-abs(x-b))


def spectrum_cos_squared(omega, phase):
    return lambda x, w=omega, p=phase: np.cos(w * freq_to_01(x) + p)**2


def spectrum_exp_cos_squared(omega, phase):
    return lambda x, w=omega, p=phase, b=1.25: np.cos(b**(w * freq_to_01(x) + p))**2


def freq_to_01(x):
    alpha = (x - MIN_L2HZ) / (MAX_L2HZ - MIN_L2HZ)
    assert ((0 <= alpha) & (alpha <= 1)).all(), "bad frequency {}".format(x)
    return alpha


def get_log_frequency_domain():
    min_l2 = np.log2(MIN_HZ)
    max_l2 = np.log2(MAX_HZ)
    l2s = np.arange(min_l2, max_l2, 1/12)
    # l2s = np.linspace(min_l2, max_l2, 5)  # debug
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
    current = 10**min_hz_order_mag
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
    labels = [str(x) if is_power_of_10(x) else "" for x in xtick_values_hz]

    # debug
    # for hz, label in zip(xtick_values_hz, labels):
    #     print("xtick {} Hz -> label {}".format(hz, repr(label)))

    return labels


def set_xticks(ax, labels=True):
    xticks_hz = get_xtick_values_hz()
    xticks_log2 = np.log2(xticks_hz)
    xtick_labels = get_xtick_labels(xticks_hz) if labels else [""] * len(xticks_hz)
    # print(xtick_labels)
    ax.set_xticks(xticks_log2)
    ax.set_xticklabels(xtick_labels)


def plot_basis_spectra():
    basis_spectra = get_basis_spectra()
    l2s = get_log_frequency_domain()
    hzs = 2**l2s
    for f in basis_spectra:
        ys = f(l2s)
        plt.plot(l2s, ys)
    ax = plt.gca()
    set_xticks(ax)
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


def normalize_spectrum_vector_to_01(spectrum_vector):
    max_val = spectrum_vector.max()
    if max_val <= 0:
        raise ValueError("spectrum vector with no positive values")
    min_val = spectrum_vector.min()
    if min_val < 0:
        raise ValueError("spectrum vector with negative values")
    if min_val == max_val:
        raise ValueError("spectrum vector with no range cannot be normalized to 01")
    f = lambda x: (spectrum_vector - min_val) / (max_val - min_val)
    res = f(spectrum_vector)
    # input("check correct individual spectrum normalization: {} , {}".format(res.min(), res.max()))
    return res


def normalize_spectrum_vectors_to_01(spectra):
    n_spectra, spec_len = spectra.shape
    res = []
    for spec_i in range(n_spectra):
        spectrum = spectra[spec_i, :]
        res.append(normalize_spectrum_vector_to_01(spectrum))
    return np.array(res)


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
    res = []
    for s in spectra:
        noisy_spectrum = get_noisy_spectrum(average_amplitude=noise_average_amplitude)
        s2 = (s+noisy_spectrum)/2
        res.append(s2)
    return np.array(res)


def get_noisy_spectrum(average_amplitude=1):
    # something like red noise (low frequencies common) with some fluctuation
    l2s = get_log_frequency_domain()
    f = spectrum_spike(mu=np.log2(MIN_HZ), dx=np.log2(MAX_HZ/MIN_HZ))
    factors = abs(np.random.normal(1,0.5,(len(l2s),)))
    spectrum = f(l2s) * factors
    spectrum *= average_amplitude / spectrum.mean()
    return spectrum


def get_random_spectrum(average_amplitude=1):
    length = len(get_basis_spectra())
    vector = np.random.uniform(0, 1, (length,))
    return get_spectrum_from_vector_in_basis(vector, average_amplitude=average_amplitude)


def get_zero_spectrum(dimensions=1):
    length = len(get_basis_spectra())
    vector = np.zeros((length,))
    spec = get_spectrum_from_vector_in_basis(vector, average_amplitude=0)
    if dimensions == 1:
        return spec.reshape((spec.size,))
    elif dimensions == 2:
        return spec.reshape((1, spec.size,))
    else:
        raise ValueError("bad dimensions {}".format(dimensions))


def get_random_spectrum_sequence(n_spectra, frames_per_spectrum, average_amplitude=1):
    initial_spectra = np.array([get_random_spectrum(average_amplitude=average_amplitude) for i in range(n_spectra)])
    # interpolate linearly between neighboring spectra, also flank by zero-intensity spectra if you want
    # initial_spectra = add_flanking_zero_spectra(initial_spectra)
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


def add_flanking_zero_spectra(spectra):
    z = get_zero_spectrum(dimensions=2)
    spectra = np.concatenate([z, spectra, z])
    return spectra


def get_articulators():
    n_spike_articulators = 0
    n_normal_articulators = 0
    n_exp_cos_squared_articulators = 8
    res = []
    res += [SpikeArticulator.in_default_position() for i in range(n_spike_articulators)]
    res += [NormalArticulator.in_default_position() for i in range(n_normal_articulators)]

    for i in range(n_exp_cos_squared_articulators):
        start_pi_factor = 1
        step_pi_factor = 1/2
        omega_min = (start_pi_factor * np.pi) + i*(step_pi_factor * np.pi)
        omega_max = ((start_pi_factor + step_pi_factor) * np.pi) + i*(step_pi_factor * np.pi)
        art = ExpCosSquaredArticulator.in_default_position(omega_min, omega_max)
        res.append(art)

    return res


def get_default_position_vector(articulators, dimensions=1):
    res = None
    for art in articulators:
        default_pos = type(art).in_default_position()
        v = default_pos.vector()
        if res is None:
            res = v
        else:
            res = np.concatenate([res, v])
    if dimensions == 1:
        assert res.shape == (res.size,)
        return res
    elif dimensions == 2:
        return res.reshape((1, res.size))
    else:
        raise ValueError("bad dimensions {}".format(dimensions))


def get_random_articulation_vector(articulators, dimensions=1):
    # this vector encodes information about how each articulator is positioned
    res = None
    for articulator in articulators:
        art = type(articulator).random()
        v = art.vector()
        if res is None:
            res = v
        else:
            res = np.concatenate([res, v])
    if dimensions == 1:
        return res.reshape((res.size,))
    elif dimensions == 2:
        return res.reshape((1, res.size,))
    else:
        raise ValueError("bad dimensions {}".format(dimensions))


def get_random_articulation_vectors(articulators, n_vectors):
    res = [get_random_articulation_vector(articulators) for i in range(n_vectors)]
    return np.array(res)


def get_spectrum_from_vector_in_articulation(vector, articulators):
    l2s = get_log_frequency_domain()
    res = np.zeros((len(l2s),))
    for art in articulators:
        n_params = art.n_params()
        sub_vector = vector[:n_params]
        vector = vector[n_params:]
        extra_init_kwargs = art.extra_init_kwargs()
        art = type(art).from_vector(sub_vector, **extra_init_kwargs)
        spectrum = art.get_spectrum()
        res += spectrum
    return res


def get_spectra_from_vectors_in_articulation(vectors, articulators, frames_per_vector, normalize_01=True):
    # move alpha through articulator space NOT spectrum space
    # so we want the articulator to have to move its mu value from a to b, for example
    # rather than just a pointwise linear combination of the spectra, which will not get the pitch moving continuously from a to b
    # interpolate linearly between neighboring articulator positions, also flank by default position vectors (articulatory setting)
    # DON'T add flanking default articulations here, this is just articulation -> spectra conversion, the flanks should be added elsewhere if you want them
    res = []
    for v0, v1 in zip(vectors[:-1], vectors[1:]):
        for a in range(frames_per_vector):
            alpha = a/frames_per_vector
            v = (1-alpha) * v0 + alpha * v1
            spectrum = get_spectrum_from_vector_in_articulation(v, articulators)
            res.append(spectrum)

    # now add the last one
    final_spectrum = get_spectrum_from_vector_in_articulation(vectors[-1], articulators)
    res.append(final_spectrum)
    assert len(res) == 1 + frames_per_vector * (len(vectors) - 1)
    res = np.array(res)
    if normalize_01:
        # normalizing on an individual-vector basis rather than the whole full vector sequence at once will allow every stage to span the interval [0,1] in intensity, rather than middle stages being too quiet because they have greater dispersion across frequency domain
        res = normalize_spectrum_vectors_to_01(res)
        # input("check correct spectrum sequence normalization: {} , {}".format(res.min(), res.max()))
    return res


def add_flanking_default_articulations(articulation_vectors, articulators):
    z = get_default_position_vector(articulators, dimensions=2)
    assert type(articulation_vectors) is np.ndarray, type(articulation_vectors)
    # initial_vectors = [z] + vectors + [z]  # wrong, it adds them elementwise, I want instead a sequence of [z, v1, v2, ..., vn, z]
    articulation_vectors = np.concatenate([z, articulation_vectors, z])  # this creates the sequence [z, v1, v2, ..., vn, z] instead of elementwise addition
    return articulation_vectors


def plot_random_articulation():
    artv = get_random_articulation_vector(get_articulators(), dimensions=1)
    plot_articulation(artv)


def plot_articulation(articulation, show=True, title=True, color=None, ax=None, tick_labels=False):
    xs = range(len(articulation))
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(xs, articulation, color=color)
    ax.set_ylim((-0.1,1.1))
    xticks = xs
    if tick_labels:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(xticks)  # try to suppress stupid warning about fixed formatter whatever: https://github.com/matplotlib/matplotlib/issues/18848
        ax.set_xticklabels(["" for x in ax.get_xticks()])
    if title:
        ax.set_title("articulation vector")
    if show:
        plt.show()


def plot_random_spectrum():
    spectrum = get_random_spectrum()
    plot_spectrum(spectrum)


def plot_random_articulation_and_spectrum():
    articulators = get_articulators()
    artv = get_random_articulation_vector(articulators, dimensions=1)
    spectrum = get_spectrum_from_vector_in_articulation(artv, articulators)
    ax = plt.subplot(2,1,1)
    plot_articulation(artv, show=False, ax=ax)
    ax = plt.subplot(2,1,2)
    plot_spectrum(spectrum, show=False, ax=ax)
    plt.show()


def plot_random_articulation_and_spectrum_sequence():
    articulators = get_articulators()
    n_vectors = 3
    # flank = False

    artvs = get_random_articulation_vectors(articulators, n_vectors)
    # if flank:
    #     artvs = add_flanking_default_articulations(artvs, articulators)
    #     n_vectors += 2  # to account for the flanking
    specvs = get_spectra_from_vectors_in_articulation(artvs, articulators, frames_per_vector=1, normalize_01=True)
    for vi in range(n_vectors):
        artv = artvs[vi]
        specv = specvs[vi]
        title = vi == 0
        tick_labels = vi == n_vectors-1
        ax = plt.subplot(n_vectors, 2, 2*vi+1)
        plot_articulation(artv, show=False, ax=ax, title=title, tick_labels=tick_labels)
        ax = plt.subplot(n_vectors, 2, 2*vi+2)
        plot_spectrum(specv, show=False, ax=ax, title=title, tick_labels=tick_labels)
    plt.show()


def plot_spectrum(spectrum, show=True, title=True, color=None, ax=None, tick_labels=True):
    l2s = get_log_frequency_domain()
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(l2s, spectrum, color=color)
    ax.set_ylim(-0.1, 1.1)
    if title:
        ax.set_title("spectrum in log-Hz domain")  # plt.title() but ax.set_title()
    set_xticks(ax, labels=tick_labels)
    if show:
        plt.show()


def plot_spectra(spectra, equal_ylim=True):
    plt.ion()
    l2s = get_log_frequency_domain()
    plt.title("spectrum in log-Hz domain")
    max_y = max(s.max() for s in spectra)
    for i, s in enumerate(spectra):
        plt.gca().clear()
        plt.plot(l2s, s)
        plt.ylim(0, max_y)
        ax = plt.gca()
        set_xticks(ax)
        plt.draw()
        plt.pause(0.1)
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
    # wav.write_signal_to_wav(waveform, "BasisSpectraOutput.wav")
    # plot_spectrum_and_fft(spectrum, waveform)

    # convert to and from sound seems too complicated/slow for now, just do spectrum with some distortion as the input to the other agents
    # noisy_spectrum = add_noise_to_spectrum(spectrum, noise_average_amplitude=0.5)
    # noisy_waveform = convert_spectrum_to_waveform(noisy_spectrum, seconds=1)
    # signal = np.concatenate([waveform, noisy_waveform])
    # plot_spectra([spectrum, noisy_spectrum])

    articulators = get_articulators()
    # articulation_vector = get_random_articulation_vector(articulators)
    # articulation_vectors = get_random_articulation_vectors(articulators, n_vectors=4)
    # spectrum = get_spectrum_from_vector_in_articulation(articulation_vector, articulators)
    # spectra = get_spectra_from_vectors_in_articulation(articulation_vectors, articulators, frames_per_vector=20)
    # spectra = add_noise_to_spectra(spectra, noise_average_amplitude=0.5)
    # plot_spectrum(spectrum)
    # plot_spectra(spectra)
    # signal = convert_spectrum_to_waveform(spectrum, seconds=1)
    # signal = convert_spectrum_sequence_to_waveform(spectra, seconds=3)
    # wav.write_signal_to_wav(signal, "BasisSpectraOutput.wav")
    # plot_spectra(spectra)

    # plot_random_articulation()
    # plot_random_articulation_and_spectrum()
    plot_random_articulation_and_spectrum_sequence()

