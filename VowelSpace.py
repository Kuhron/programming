import numpy as np
import math
import random
import matplotlib.pyplot as plt


class Exemplar:
    def __init__(self, sound, articulation, meaning):
        self.sound = sound
        self.articulation = articulation
        self.meaning = meaning


class DistortionFunction01:
    def __init__(self, frequency, amplitude):
        assert frequency > 0
        assert frequency % 1 == 0
        assert -1 <= amplitude <= 1
        self.frequency = frequency
        self.amplitude = amplitude

    def __call__(self, x):
        a = self.amplitude
        n = self.frequency
        n_pi = n * np.pi
        return x + a * 1/n_pi * np.sin(n_pi * x)
        # maps [0,1] interval to itself with some bending, still monotonic
        # so iterating different functions of this form can give you wiggly shape that is still 1-to-1 in [0,1] and monotonically increasing

    @staticmethod
    def random(stdev):
        i = 0
        while True:
            a = np.random.normal(0, stdev)
            if -1 <= a <= 1:
                break
            i += 1
            if i > 10000:
                raise RuntimeError(f"stdev {stdev} led to too many iterations when trying to roll scale parameter; please reduce stdev")
        n = random.randint(1, 10)
        return DistortionFunction01(frequency=n, amplitude=a)

    @staticmethod
    def regression_to_mean(stdev):
        # a should be between 0 and 1 so that more things map toward the middle and away from the edges
        while True:
            a = abs(np.random.normal(0, stdev))
            if 0 <= a <= 1:
                break
        n = 2  # only one period inside the box
        return DistortionFunction01(frequency=n, amplitude=a)

    def plot(self):
        xs = np.linspace(0, 1, 101)
        ys = self(xs)
        plt.plot(xs, ys)
        plt.show()

    def plot_image_distribution(self):
        # image as in the image of the function
        xs = np.linspace(0, 1, 10001)
        ys = self(xs)
        plt.hist(ys, bins=100)
        plt.show()


class DistortionFunctionSeries:
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, x):
        for f in self.funcs:
            x = f(x)
        return x

    @staticmethod
    def random(stdev):
        n_funcs = random.randint(3, 8)
        funcs = [DistortionFunction01.random(stdev) for i in range(n_funcs)]
        return DistortionFunctionSeries(funcs)


class Mouth:
    def __init__(self, f1_distortion_func, f2_distortion_func, speech_error_stdev):
        self.f1_distortion_func = f1_distortion_func
        self.f2_distortion_func = f2_distortion_func
        self.speech_error_stdev = speech_error_stdev  # this should be relatively small e.g. 0.1 or less
        self.sound_to_exemplar_articulations = []

    @staticmethod
    def random(anatomical_stdev, speech_error_stdev):
        f1_distortion_func = DistortionFunctionSeries.random(anatomical_stdev)
        f2_distortion_func = DistortionFunctionSeries.random(anatomical_stdev)
        return Mouth(f1_distortion_func, f2_distortion_func, speech_error_stdev)

    def convert_articulation_to_formants(self, arts):
        f1_x, f2_x = arts
        f1 = self.distort_f1(f1_x)
        f2 = self.distort_f2(f2_x)
        return f1, f2

    def pronounce(self, arts, commit_to_memory=True):
        f1_x, f2_x = arts
        # slightly perturb the ARTICULATIONS as speech errors, and maybe this will have a small/large effect on the output formant (based on individual anatomy)
        f1_x_with_speech_error = self.add_speech_error(f1_x)
        f2_x_with_speech_error = self.add_speech_error(f2_x)
        produced_f1 = self.distort_f1(f1_x_with_speech_error)
        produced_f2 = self.distort_f2(f2_x_with_speech_error)
        pronunciation = (produced_f1, produced_f2)
        if commit_to_memory:
            exemplar = Exemplar(pronunciation, arts, meaning=None)
            self.add_sound_articulation_exemplar(exemplar)
        return pronunciation

    def add_speech_error(self, x):
        f_error = DistortionFunctionSeries.random(self.speech_error_stdev)
        f_regression = DistortionFunction01.regression_to_mean(self.speech_error_stdev)
        output = f_error(x)  # produce error first
        output = f_regression(output)  # add slight regression to the mean of *articulatory* displacement
        return output

    def distort_f1(self, x):
        return self.f1_distortion_func(x)

    def distort_f2(self, x):
        return self.f2_distortion_func(x)

    def plot_distortions(self):
        xs = np.linspace(0,1,101)
        f1s = self.distort_f1(xs)
        f2s = self.distort_f2(xs)
        plt.plot(xs, f1s, label="f1", c="r")
        plt.plot(xs, f2s, label="f2", c="b")
        plt.plot(xs, xs, label="y=x", c="#777777")
        plt.legend()
        plt.show()

    def plot_target_grid(self):
        f1_xs = np.arange(0.1, 1, 0.1)
        f2_xs = np.arange(0.1, 1, 0.1)
        for f1_x in f1_xs:
            for f2_x in f2_xs:
                arts = (f1_x, f2_x)
                f1, f2 = self.convert_articulation_to_formants(arts)
                plt.scatter(f1, f2, c="b")
        plt.show()

    def babble(self):
        for i in range(100):
            production_target = np.random.uniform(0, 1, (2,))
            heard_formants = self.pronounce(production_target)
            exemplar = Exemplar(heard_formants, production_target, meaning=None)
            self.add_sound_articulation_exemplar(exemplar)
            production_discrepancy = np.linalg.norm(heard_formants - production_target)
            # print(f"tried to produce {production_target}, got {heard_formants}, distance {production_discrepancy}")

    def add_sound_articulation_exemplar(self, exemplar):
        self.sound_to_exemplar_articulations.append(exemplar)
        max_exemplars = 100  # of anything whatsoever
        self.sound_to_exemplar_articulations = self.sound_to_exemplar_articulations[-max_exemplars:]

    def estimate_articulation_for_sound(self, heard_formants):
        # get nearest neighbors from self.sound_to_exemplar_articulations
        k_neighbors = 5
        nearest_neighbors = self.get_nearest_articulations_to_sound(heard_formants, k_neighbors)
        average_articulation = sum(np.array(x) for x in nearest_neighbors) / k_neighbors
        assert average_articulation.shape == (2,)
        return average_articulation

    def get_nearest_articulations_to_sound(self, heard_formants, k_neighbors):
        candidates = [x.sound for x in self.sound_to_exemplar_articulations]
        distances = {}
        for c in candidates:
            d = np.linalg.norm(np.array(heard_formants) - np.array(c))
            distances[c] = d
        ranked = sorted(distances.items(), key=lambda kv: kv[1], reverse=False)  # homebrew, going to be inefficient, use kdtree later if necessary
        top_k = ranked[:k_neighbors]
        neighbors = [tup[0] for tup in top_k]
        return neighbors


def play_repeat_after_me_game(mouths):
    for i in range(1000):
        speaker_index = random.randrange(len(mouths))
        speaker = mouths[speaker_index]
        production_target = np.random.uniform(0, 1, (2,))
        pronunciation = speaker.pronounce(production_target)
        print(f"target {production_target}, produced {pronunciation}")
        for m in mouths:
            estimated_articulation = m.estimate_articulation_for_sound(pronunciation)
            repetition = m.pronounce(estimated_articulation)
            imitation_discrepancy = np.linalg.norm(np.array(repetition) - np.array(pronunciation))
            print(f"imitation produced {repetition}, discrepancy {imitation_discrepancy}")


def play_iterated_repeat_after_me_game(mouths, n_steps, window_length, plot_ion=True):
    # start with a random production target, then iterate where each mouth targets whatever the other one just said, and plot how the productions change to see what the dynamics are like
    # they take turns going around the circle in order of index in the list `mouths`
    # window length is number of previous pronunciations to look at

    if plot_ion:
        plt.ion()
        fignum = plt.gcf().number  # use to determine if user has closed plot

    initial_sound_target = np.random.uniform(0, 1, (2,))
    previous_pronunciations = []

    sound_target_f1s = []
    sound_target_f2s = []

    for i in range(n_steps):
        if i % 100 == 0:
            print(f"iterated repeat after me game i={i}/{n_steps}")
        # speaker = mouths[i % len(mouths)]
        speaker = random.choice(mouths)

        if i == 0:
            sound_target = initial_sound_target
        else:
            sound_target = sum(previous_pronunciations) / len(previous_pronunciations)
        # print(f"current sound target is {sound_target}")
        sound_target_f1s.append(sound_target[0])
        sound_target_f2s.append(sound_target[1])

        articulation_target = speaker.estimate_articulation_for_sound(sound_target)
        pronunciation = speaker.pronounce(articulation_target, commit_to_memory=True)
        this_f1, this_f2 = pronunciation

        previous_pronunciations.append(np.array(pronunciation))
        previous_pronunciations = previous_pronunciations[-window_length:]

        if i > 0:  # can't calculate difference for i=0
            d_f1 = this_f1 - earlier_f1
            d_f2 = this_f2 - earlier_f2

            if plot_ion:
                plt.gcf().clear()
                plt.arrow(earlier_f1, earlier_f2, d_f1, d_f2)
                plt.scatter(this_f1, this_f2)
                plt.xlim(0,1)
                plt.ylim(0,1)
                plt.draw()
                plt.pause(0.01)

        earlier_f1 = this_f1
        earlier_f2 = this_f2

        if plot_ion and not plt.fignum_exists(fignum):
            print("user closed plot; exiting")
            break
    if plot_ion:
        plt.ioff()

    # plot how the target has varied over the course of the game
    plt.plot(sound_target_f1s, c="r", label="target_f1", alpha=0.7)
    plt.plot(sound_target_f2s, c="b", label="target_f2", alpha=0.7)
    plt.legend()
    plt.ylim(0,1)
    plt.show()


def plot_how_mouths_say_articulation(mouths):
    articulation = np.random.uniform(0, 1, (2,))
    for m in mouths:
        fs = m.convert_articulation_to_formants(articulation)
        plt.scatter(*fs)
    plt.scatter(*articulation, marker="x")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()


def play_classification_game(mouths, n_steps):
    for i in range(n_steps):
        card_color = random.choice(["red", "yellow", "blue"])


if __name__ == "__main__":
    anatomical_stdev = 0.05
    speech_error_stdev = 0.05
    # observations:
    # - higher speech error leads to collapse to stable points (the four corners) even in absence of anatomical variation
    # - with forgetting of exemplar pronunciations, drift can occur even with no anatomical OR speech-error variation (happens in discrete jumps)
    # - small anatomical stdev and small speech error stdev (e.g. both 0.01) leads to slow drift, which can either converge or persist throughout

    n_mouths = 4
    n_steps = 10000
    window_length = 4
    mouths = [Mouth.random(anatomical_stdev, speech_error_stdev) for i in range(n_mouths)]
    # m1.plot_distortions()
    # m1.plot_target_grid()

    for m in mouths:
        m.babble()

    # plot_how_mouths_say_articulation()
    # play_repeat_after_me_game(mouths)
    play_iterated_repeat_after_me_game(mouths, n_steps, window_length, plot_ion=False)
