import random
import time

import MIDI as midi
import numpy as np

import Music.WavUtil as wav


EPSILON = 1e-9
MIDI_MIN_LOUDNESS = 0
MIDI_MAX_LOUDNESS = 127


class Note:
    def __init__(self, name, duration):
        self.name = name
        assert self.name[1:] in OCTAVES, "invalid name {}".format(self.name)
        self.pitch_class = self.name[0]
        assert is_pitch_class(self.pitch_class)
        self.octave = int(self.name[1])
        self.pitch_number = get_pitch_number_from_note_name(self.name)
        self.midi_pitch_number = get_midi_pitch_number_from_note_name(self.name)
        self.frequency = get_frequency_from_pitch_number(self.pitch_number)
        self.duration = duration
        self.midi_loudness = int(MIDI_MAX_LOUDNESS / 3)

    def get_wav_signal(self, truncate=True, spectrum=None):
        return wav.get_signal_from_freq(self.frequency, self.duration.duration_seconds, initial_click=False, truncate=truncate, spectrum=spectrum)

    def output_to_midi(self, midi_output, stop=True):
        midi_output.note_on(self.midi_pitch_number, self.midi_loudness)
        if stop:
            self.stop_output_to_midi(midi_output)

    def stop_output_to_midi(self, midi_output, sleep=True):
        if sleep:
            time.sleep(self.duration.duration_seconds)
        midi_output.note_off(self.midi_pitch_number, self.midi_loudness)

    def __add__(self, other):
        if type(other) is Interval:
            new_name = add_interval_to_note_name(self.name, other)
            return Note(new_name, self.duration)
        elif type(other) is Note:
            new_names = [self.name, other.name]
            return Chord(new_names, self.duration)
        elif type(other) is Chord:
            new_names = [self.name] + other.names
            return Chord(new_names, self.duration)
        return NotImplemented

    def __sub__(self, other):
        if type(other) is Interval:
            neg = Interval(-other.step)
            return self + neg
        return NotImplemented

    def __repr__(self):
        return "{0}_{1}".format(self.name, self.duration)


class Chord:
    def __init__(self, names, duration):
        self.names = list(np.unique(names))
        self.duration = duration
        self.notes = [Note(name, duration) for name in self.names]

    def get_wav_signal(self):
        return sum(note.get_wav_signal(truncate=False) for note in self.notes)

    def output_to_midi(self, midi_output):
        for note in self.notes:
            note.output_to_midi(midi_output, stop=False)
        for i, note in enumerate(self.notes):
            sleep = i == 0
            note.stop_output_to_midi(midi_output, sleep=sleep)

    def __add__(self, other):
        if type(other) is Interval:
            new_names = [add_interval_to_note_name(name, other) for name in self.names]
            return Chord(new_names, self.duration)
        elif type(other) is Note:
            new_names = self.names + [other.name]
            return Chord(new_names, self.duration)
        elif type(other) is Chord:
            new_names = self.names + other.names
            return Chord(new_names, self.duration)
        return NotImplemented

    def __sub__(self, other):
        if type(other) is Interval:
            neg = Interval(-other.step)
            return self + neg
        return NotImplemented

    def __repr__(self):
        return "+".join(sorted(repr(x) for x in self.notes))


class Rest:
    def __init__(self, duration):
        self.duration = duration

    def get_wav_signal(self):
        return wav.get_silence_for_duration(self.duration.duration_seconds)

    def output_to_midi(self, midi_output, stop=True):
        time.sleep(self.duration.duration_seconds)
        # stop is irrelevant

    def __repr__(self):
        return "r_{0}".format(self.duration)


class Duration:
    def __init__(self, duration_beats, tempo_bpm):
        self.duration_beats = duration_beats
        self.tempo_bpm = tempo_bpm
        self.duration_seconds = self.duration_beats * 60 / self.tempo_bpm

    def __repr__(self):
        return "{0}@{1}bpm".format(self.duration_beats, self.tempo_bpm)


class Interval:
    def __init__(self, step):
        self.step = step


class Scale:
    instances = []

    def __init__(self, steps, base=60):
        self.instances.append(self)
        self.base = base
        self.steps = steps
        self.cumulative_steps = [0] + [sum(self.steps[:i]) for i in range(1, len(self.steps) + 1)]
        assert sum(steps) == 12, "step sizes must sum to 12"

    def get_pitch(self, n):
        octave = n // len(self.steps)
        index = n % len(self.steps)
        return self.base + (12 * octave) + self.cumulative_steps[index]

    def set_base(self, base):
        self.base = base

    def __len__(self):
        return len(self.steps)

    def __eq__(self, other):
        if type(other) is not Scale:
            return NotImplemented
        return self.steps == other.steps


class Measure:
    def __init__(self, n_beats, scale, resolution, last_pitch_index, overhanging_duration_from_previous=0):
        self.n_beats = n_beats
        self.scale = scale
        self.resolution = resolution
        self.overhanging_duration_from_previous = overhanging_duration_from_previous if overhanging_duration_from_previous > EPSILON else 0
        self.duration_buffer = 0
        self.durations = self.get_durations(self.resolution, self.overhanging_duration_from_previous)
        self.times = self.get_times()
        self.pitches = self.get_pitches(last_pitch_index)
        self.notes = self.get_notes()

    def get_durations(self, resolution, overhanging_duration_from_previous):
        self.overhanging_duration_from_previous = overhanging_duration_from_previous
        possible_durations = np.arange(resolution, self.n_beats + EPSILON, resolution)
        durations = []
        
        current_time = overhanging_duration_from_previous
        while current_time < self.n_beats:
            duration_left = self.n_beats - current_time
            if current_time == 0:
                weights = [1 / max(1, i) if i != 1 else 2.5 for i in possible_durations]
            else:
                weights = [1 / max(1, i) if i != 1 else 1.5 if i <= duration_left else 0 for i in possible_durations]
                # weights = [1 if i == resolution else 0 for i in possible_durations]
            norm_weights = [i / sum(weights) for i in weights]
            duration = np.random.choice(possible_durations, p=norm_weights)
            durations.append(duration)
            current_time += duration
        return durations

    def get_pitches(self, last_pitch_index):
        max_pitch_index_deviation = len(self.scale) * 2
        pitch_indices = []
        pitch_index = last_pitch_index if last_pitch_index is not None else 0

        for t in self.times:
            if t % self.n_beats < EPSILON and random.random() < 0.25:
                pitch_index = (pitch_index // len(self.scale)) * len(self.scale)
            else:
                pitch_step = random.choice([-2, -1, 1, 2])
                pitch_index += pitch_step

            if pitch_index >= max_pitch_index_deviation:
                pitch_index -= len(self.scale)
            elif pitch_index <= -max_pitch_index_deviation:
                pitch_index += len(self.scale)

            pitch_indices.append(pitch_index)

        # pitch_indices = [random.randint(-6, 6) for _ in range(len(self.durations))]
        self.pitch_indices = pitch_indices
        return [self.scale.get_pitch(i) for i in pitch_indices]

    def get_times(self):
        # times at beginning of notes
        times = [self.overhanging_duration_from_previous + sum(self.durations[:i]) for i in range(0, len(self.durations))]
        # times = [(t - t % self.resolution) if t % self.resolution < EPSILON else t for t in times]
        return times

    def get_notes(self):
        result = []
        for t, duration, pitch in zip(self.times, self.durations, self.pitches):
            channel = 1
            loudness = 96
            result.append(["note", t, duration - self.duration_buffer, channel, pitch, loudness])
        return result

    def get_last_pitch_index(self):
        return self.pitch_indices[-1]

    def get_overhanging_duration_to_next(self):
        return max(0, self.times[-1] + self.durations[-1] - self.n_beats)


def get_scale_from_color_pair(color1, color2):
    assert type(color1) is list and type(color2) is list

    sum1 = sum(color1)
    sum2 = sum(color2)
    assert 4 <= sum1 <= 6 and 4 <= sum2 <= 6

    begin1 = 0
    end1 = sum1
    begin2 = 12 - sum2
    end2 = 12
    gap = begin2 - end1

    if gap < 0:
        # first color takes precedence
        intervals = color1 + color2[1:]
    elif gap == 0:
        intervals = color1 + color2
    elif gap < 4:
        intervals = color1 + [gap] + color2
    elif gap == 4:
        intervals = color1 + [2, 2] + color2
    else:
        raise ValueError("something went wrong when creating a scale from these color intervals: {0}, {1}".format(color1, color2))

    assert sum(intervals) == 12, ("something went wrong when creating a scale from these color intervals: {0}, {1}".format(color1, color2))

    return Scale(intervals)


def get_pitch_number_from_note_name(s):
    # C4 = middle C (first C below 440 Hz) is 0
    if s is None:
        return None

    if any([i not in "AMRBCKDHEFXGLJ#b0123456789" for i in s]) or len(s) > 3:
        raise ValueError("Invalid note name {0}".format(s))

    n = s[:1]
    o = s[1:]
    v = pitch_class_to_number(n[0])
    if len(n) > 1:
        assert not is_black_key(n[0])
        v += (1 if n[1] == "#" else -1 if n[1] == "b" else ValueError)

    o = int(o) - 4
    v += 12 * o
    return v


def get_midi_pitch_number_from_note_name(s):
    return 60 + get_pitch_number_from_note_name(s)


def get_note_name_from_pitch_number(n):
    # increment octave between B and C, according to https://en.wikipedia.org/wiki/Scientific_pitch_notation
    # (//) rounds toward negative infinity, true floor division
    o, v = divmod(n, 12)
    pitch_class = number_to_pitch_class(v)
    octave = o + 4
    return pitch_class + str(octave)


def get_frequency_from_pitch_number(n):
    # pitch number is offset from middle C
    A4_offset = -9 + n
    return int(440 * 2**(A4_offset * 1.0/12))


def get_frequency_from_note_name(s, shift=0):
    v = get_pitch_number_from_note_name(s)
    v += shift
    return get_frequency_from_pitch_number(v)


def add_interval_to_note_name(name, interval):
    n = get_pitch_number_from_note_name(name)
    n += interval.step
    return get_note_name_from_pitch_number(n)


def is_black_key(pitch_class):
    return PITCH_CLASS_TO_NUMBER[pitch_class] in [1, 3, 6, 8, 10]


def is_pitch_class(x):
    return x in PITCH_CLASS_TO_NUMBER


def pitch_class_to_number(x):
    return PITCH_CLASS_TO_NUMBER[x]


def number_to_pitch_class(x):
    return NUMBER_TO_PITCH_CLASS[x]


PITCH_CLASS_TO_NUMBER = {
    "C": 0,
    "K": 1,
    "D": 2,
    "H": 3,
    "E": 4,
    "F": 5,
    "X": 6,
    "G": 7,
    "L": 8, "J": 8,
    "A": 9,
    "M": 10, "R": 10,
    "B": 11,
}
NUMBER_TO_PITCH_CLASS = "CKDHEFXGJARB"
MIN_OCTAVE = 1
MAX_OCTAVE = 8
OCTAVES = [str(i) for i in range(MIN_OCTAVE, MAX_OCTAVE + 1)]


class INTERVALS:
    unison = Interval(0)
    minor_second = Interval(1)
    major_second = Interval(2)
    minor_third = Interval(3)
    major_third = Interval(4)
    fourth = Interval(5)
    perfect_fourth = fourth
    augmented_fourth = Interval(6)
    diminished_fifth = augmented_fourth
    fifth = Interval(7)
    minor_sixth = Interval(8)
    major_sixth = Interval(9)
    minor_seventh = Interval(10)
    major_seventh = Interval(11)
    octave = Interval(12)


class COLOR_INTERVALS:
    gold = [1, 2, 1]
    red = [1, 2, 2]
    orange = [1, 2, 3]
    yellow = [1, 3, 1]
    green = [1, 3, 2]
    blue = [2, 1, 2]
    purple = [2, 1, 3]
    white = [2, 2, 1]
    silver = [2, 2, 2]
    black = [3, 1, 2]
    pink = [3, 1, 1]


class SCALES:
    chromatic = Scale([1] * 12)

    mayamalavagowla = Scale([1, 3, 1, 2, 1, 3, 1])
    assert get_scale_from_color_pair(COLOR_INTERVALS.yellow, COLOR_INTERVALS.yellow) == mayamalavagowla
    ionian = Scale([2, 2, 1, 2, 2, 2, 1])
    assert get_scale_from_color_pair(COLOR_INTERVALS.white, COLOR_INTERVALS.white) == ionian
    dorian = Scale([2, 1, 2, 2, 2, 1, 2])
    assert get_scale_from_color_pair(COLOR_INTERVALS.blue, COLOR_INTERVALS.blue) == dorian
    phrygian = Scale([1, 2, 2, 2, 1, 2, 2])
    assert get_scale_from_color_pair(COLOR_INTERVALS.red, COLOR_INTERVALS.red) == phrygian
    lydian = Scale([2, 2, 2, 1, 2, 2, 1])
    assert get_scale_from_color_pair(COLOR_INTERVALS.silver, COLOR_INTERVALS.white) == lydian
    mixolydian = Scale([2, 2, 1, 2, 2, 1, 2])
    assert get_scale_from_color_pair(COLOR_INTERVALS.white, COLOR_INTERVALS.blue) == mixolydian
    aeolian = Scale([2, 1, 2, 2, 1, 2, 2])
    assert get_scale_from_color_pair(COLOR_INTERVALS.blue, COLOR_INTERVALS.red) == aeolian
    locrian = Scale([1, 2, 2, 1, 2, 2, 2])
    assert get_scale_from_color_pair(COLOR_INTERVALS.red, COLOR_INTERVALS.silver) == locrian
    wangandar = Scale([2, 1, 3, 1, 2, 1, 2])
    assert get_scale_from_color_pair(COLOR_INTERVALS.purple, COLOR_INTERVALS.blue) == wangandar
    phrygian_dominant = Scale([1, 3, 1, 2, 1, 2, 2])
    assert get_scale_from_color_pair(COLOR_INTERVALS.yellow, COLOR_INTERVALS.red) == phrygian_dominant
    black_pink = Scale([3, 1, 2, 1, 3, 1, 1])
    assert get_scale_from_color_pair(COLOR_INTERVALS.black, COLOR_INTERVALS.pink) == black_pink
    harmonic_minor = Scale([2, 1, 2, 2, 1, 3, 1])
    assert get_scale_from_color_pair(COLOR_INTERVALS.blue, COLOR_INTERVALS.yellow) == harmonic_minor

    octatonic_major = Scale([2, 1, 2, 1, 2, 1, 2, 1])
    octatonic_minor = Scale([1, 2, 1, 2, 1, 2, 1, 2])
    whole_tone = Scale([2, 2, 2, 2, 2, 2])

    pure_gold = get_scale_from_color_pair(COLOR_INTERVALS.gold, COLOR_INTERVALS.gold)
    pure_red = phrygian
    pure_orange = get_scale_from_color_pair(COLOR_INTERVALS.orange, COLOR_INTERVALS.orange)
    pure_yellow = mayamalavagowla
    pure_green = get_scale_from_color_pair(COLOR_INTERVALS.green, COLOR_INTERVALS.green)
    pure_blue = dorian
    pure_purple = get_scale_from_color_pair(COLOR_INTERVALS.purple, COLOR_INTERVALS.purple)
    pure_white = ionian
    pure_silver = whole_tone
    pure_black = get_scale_from_color_pair(COLOR_INTERVALS.black, COLOR_INTERVALS.black)
    pure_pink = get_scale_from_color_pair(COLOR_INTERVALS.pink, COLOR_INTERVALS.pink)
