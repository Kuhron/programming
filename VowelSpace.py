import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


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
    def __init__(self, f1_distortion_func, f2_distortion_func, speech_error_stdev, name=None):
        self.name = name
        self.f1_distortion_func = f1_distortion_func
        self.f2_distortion_func = f2_distortion_func
        self.speech_error_stdev = speech_error_stdev  # this should be relatively small e.g. 0.1 or less
        self.exemplars = []
        self.kdtree = None

    @staticmethod
    def random(anatomical_stdev, speech_error_stdev, **kwargs):
        f1_distortion_func = DistortionFunctionSeries.random(anatomical_stdev)
        f2_distortion_func = DistortionFunctionSeries.random(anatomical_stdev)
        return Mouth(f1_distortion_func, f2_distortion_func, speech_error_stdev, **kwargs)

    def __repr__(self):
        return self.name

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
            self.add_exemplar(exemplar)
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
            articulation_target = np.random.uniform(0, 1, (2,))
            heard_formants = self.pronounce(articulation_target)
            exemplar = Exemplar(heard_formants, articulation_target, meaning=None)
            self.add_exemplar(exemplar)
            production_discrepancy = np.linalg.norm(heard_formants - articulation_target)
            # print(f"tried to produce {production_target}, got {heard_formants}, distance {production_discrepancy}")

    def get_random_pronunciation(self):
        articulation_target = np.random.uniform(0, 1, (2,))
        heard_formants = self.pronounce(articulation_target)
        return heard_formants

    def add_exemplar(self, exemplar):
        self.invalidate_kdtree()
        self.exemplars.append(exemplar)
        max_exemplars = 100  # of anything whatsoever
        self.exemplars = self.exemplars[-max_exemplars:]

    def estimate_articulation_for_sound(self, heard_formants):
        # get nearest neighbors from self.exemplars
        k_neighbors = 5
        nearest_neighbors = self.get_nearest_articulations_to_sound(heard_formants, k_neighbors)
        average_articulation = sum(np.array(x) for x in nearest_neighbors) / k_neighbors
        assert average_articulation.shape == (2,)
        return average_articulation

    def invalidate_kdtree(self):
        self.kdtree = None

    def create_kdtree(self):
        positions_in_sound_space = [x.sound for x in self.exemplars]
        self.kdtree = KDTree(positions_in_sound_space)

    def get_nearest_exemplars_to_sound(self, heard_formants, k_neighbors):
        if self.kdtree is None:  # need to rebuild it because it was invalidated due to adding or removing exemplars
            self.create_kdtree()
        neighbor_distances, neighbor_indices = self.kdtree.query(heard_formants, k_neighbors)
        exemplars = [self.exemplars[i] for i in neighbor_indices]
        return exemplars

    def get_nearest_articulations_to_sound(self, heard_formants, k_neighbors):
        exemplars = self.get_nearest_exemplars_to_sound(heard_formants, k_neighbors)
        return [x.articulation for x in exemplars]

    def pronounce_meaning(self, meaning, window_length):
        exemplars = [x for x in self.exemplars if x.meaning == meaning]
        exemplars = exemplars[-window_length:]  # only look at the most recent n exemplars with this meaning
        if len(exemplars) == 0:
            return self.get_random_pronunciation()
        # pronunciations = [x.sound for x in exemplars]
        # suppose that the speaker uses articulation examples rather than sound examples to pronounce the word again
        articulations = [x.articulation for x in exemplars]
        articulation_target = sum(articulations) / len(articulations)
        return self.pronounce(articulation_target, commit_to_memory=True)

    def predict_meaning_from_sound(self, sound):
        k_neighbors = 5
        nearest_neighbors = self.get_nearest_exemplars_to_sound(sound, k_neighbors)
        meanings = [x.meaning for x in nearest_neighbors]
        mode_meaning = max(meanings, key=meanings.count)
        return mode_meaning

    def plot_color_meanings_in_sound_space(self):
        f1s = np.linspace(0, 1, 26)
        f2s = np.linspace(0, 1, 26)
        # assumes the meaning is a color string
        for f1 in f1s:
            # print(f1)
            for f2 in f2s:
                color = self.predict_meaning_from_sound((f1, f2))
                if color is not None:
                    plt.scatter(f1, f2, c=color)
        plt.title(f"color meanings for {self.name}")
        plt.show()


def play_repeat_after_me_game(mouths):
    for i in range(1000):
        speaker_index = random.randrange(len(mouths))
        speaker = mouths[speaker_index]
        articulation_target = np.random.uniform(0, 1, (2,))
        pronunciation = speaker.pronounce(articulation_target)
        print(f"target {articulation_target}, produced {pronunciation}")
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


def compare_meaning_agreement(mouths, possible_meanings):
    n_meanings = len(possible_meanings) + 1  # since None is a possible meaning
    dispersions = []
    for i in range(1000):
        sound = np.random.uniform(0, 1, (2,))
        meanings = [m.predict_meaning_from_sound(sound) for m in mouths]
        dispersion = len(set(meanings))
        dispersions.append(dispersion)
    dispersions = np.array(dispersions)
    average_dispersion = np.mean(dispersions)
    consensuses = 1/dispersions
    average_consensus = np.mean(consensuses)
    print(f"average dispersion {average_dispersion}, average consensus {average_consensus}")


def play_classification_game(mouths, n_steps, window_length, colors):
    accuracies = {m.name: [] for m in mouths}
    for i in range(n_steps):
        if i % 100 == 0:
            print(f"classification game step {i}/{n_steps}")
        color = random.choice(colors)
        speaker = random.choice(mouths)
        # speaker must describe the color
        pronunciation = speaker.pronounce_meaning(color, window_length)
        for m in mouths:
            predicted_color = m.predict_meaning_from_sound(pronunciation)
            # print(f"{m} predicted {predicted_color} for {color}")
            correct = int(predicted_color == color)
            accuracies[m.name].append(correct)

            # if it was correct, change nothing (it will reinforce itself by being a new exemplar)
            # if it was incorrect, do what? I think it should be a self-reinforcing mechanism, right? since the new exemplars will help move the boundaries of the categories

            articulation = m.estimate_articulation_for_sound(pronunciation)
            exemplar = Exemplar(pronunciation, articulation, meaning=color)
            m.add_exemplar(exemplar)
        # print(f"{color} was pronounced as {pronunciation}")
    # need to add prediction/learning

    for m in mouths:
        correctness_array = accuracies[m.name]
        # cumsum = np.cumsum(correctness_array)
        moving_window_n = 100
        accuracy_proportions = np.convolve(correctness_array, np.ones(moving_window_n)/moving_window_n, mode='valid')
        # accuracy_proportions = cumsum / (1+ np.arange(len(cumsum)))
        plt.plot(accuracy_proportions, label=m.name)
    chance_accuracy = 1/len(colors)
    plt.plot(range(len(accuracy_proportions)), [chance_accuracy]*len(accuracy_proportions), c="k", label="monkey")
    plt.legend()
    plt.show()

    for m in mouths:
        m.plot_color_meanings_in_sound_space()



if __name__ == "__main__":
    anatomical_stdev = 0.1
    speech_error_stdev = 0.1
    # observations:
    # - higher speech error leads to collapse to stable points (the four corners) even in absence of anatomical variation
    # - with forgetting of exemplar pronunciations, drift can occur even with no anatomical OR speech-error variation (happens in discrete jumps)
    # - small anatomical stdev and small speech error stdev (e.g. both 0.01) leads to slow drift, which can either converge or persist throughout

    n_mouths = 4
    n_steps = 1000
    window_length = 10
    mouths = [Mouth.random(anatomical_stdev, speech_error_stdev, name=f"M{i}") for i in range(n_mouths)]
    # m1.plot_distortions()
    # m1.plot_target_grid()

    for m in mouths:
        m.babble()

    # plot_how_mouths_say_articulation()
    # play_repeat_after_me_game(mouths)
    # play_iterated_repeat_after_me_game(mouths, n_steps, window_length, plot_ion=False)
    
    colors = ["red", "orange", "yellow", "green", "blue"] #, "purple", "brown", "black", "white"]
    play_classification_game(mouths, n_steps, window_length, colors)
    compare_meaning_agreement(mouths, colors)
