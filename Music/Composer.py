import random

import MIDI as midi
import numpy as np

import Music.MusicalStructureUtil as structure


def get_notes_original(n, scale=structure.SCALES.chromatic, tempo_factor=50):
	result = []
	current_time = 0
	last_duration = 0
	pitch_index = 0
	duration_buffer = 0.05
	max_pitch_index_deviation = len(scale) * 2
	n_beats = random.choice([3, 3, 4, 4, 4, 5, 6, 7])
	base = random.choice([60 + i for i in range(-12, 13)])
	scale.set_base(base)
	for i in range(n):
		d_time = last_duration + duration_buffer
		current_time += d_time
		
		if current_time % n_beats < 0.01:
			current_time = (current_time // n_beats) * n_beats
			if random.random() < 0.5:
				duration = 1
			else:
				duration = random.choice([0.5, 1])
			if random.random() < 0.25:
				pitch_index = (pitch_index // len(scale)) * len(scale)
			else:
				pitch_step = random.choice([-2, -1, 1, 2])
				pitch_index += pitch_step
		else:
			max_duration = n_beats - (current_time % n_beats)
			if current_time % 1 == 0:
				duration = random.choice([0.5, 1])
			elif current_time % n_beats < n_beats - 0.5:
				duration = random.choice([0.5, 0.5, 0.5, 1])
			else:
				duration = 0.5
			pitch_step = random.choice([-2, -1, 1, 2])
			pitch_index += pitch_step

		if pitch_index >= max_pitch_index_deviation:
			pitch_index -= len(scale)
		elif pitch_index <= -max_pitch_index_deviation:
			pitch_index += len(scale)

		duration = duration - duration_buffer
		channel = 1
		pitch = scale.get_pitch(pitch_index)
		loudness = 96
		result.append(["note", current_time, duration, channel, pitch, loudness])
		last_duration = duration

	for note in result:
		note[1] *= tempo_factor
		note[2] *= tempo_factor
	return result


def get_notes_with_measures(n_measures, n_beats, resolution, scale, tempo_factor):
	result = []
	measures = []
	transform_time = lambda t, m: ((n_beats * m) + t) * tempo_factor
	transform_duration = lambda d: d * tempo_factor

	for m in range(n_measures):
		if m == 0:
			last_pitch_index = None
			overhanging_duration_from_previous = 0
		else:
			last_pitch_index = measures[-1].get_last_pitch_index()
			overhanging_duration_from_previous = measures[-1].get_overhanging_duration_to_next()
		measure = Measure(n_beats, scale, resolution, last_pitch_index, overhanging_duration_from_previous)
		measures.append(measure)
		notes = measure.notes
		for note in notes:
			note[1] = transform_time(note[1], m)
			note[2] = transform_duration(note[2])
		result.extend(notes)

	return result


def get_colorful_walk(n_notes, scales, tempo_factor):
	notes = []
	transform_time = lambda t, m: ((n_notes * m) + t) * tempo_factor
	transform_duration = lambda d: d * tempo_factor

	pitches = []  # for keeping track of last n pitches to get new scale
	pitch_indices = []
	current_scale = random.choice(scales)

	for t in range(n_notes):
		channel = 1
		loudness = 96
		duration = 0.5

		if len(pitch_indices) == 0:
			pitch_index = random.randint(-10, 10)
		else:
			pitch_index = pitch_indices[-1] + random.choice([-4, -3, -2, -1, 1, 2, 3, 4])

		if pitch_index >= len(current_scale) * 2:
			pitch_index -= len(current_scale)
		elif pitch_index <= -len(current_scale) * 2:
			pitch_index += len(current_scale)

		pitch = current_scale.get_pitch(pitch_index)
		pitch_indices.append(pitch_index)
		pitches.append(pitch)

		if random.random() < 0.9:  # can replace this with better logic to take last n notes and switch to the key/mode they "suggest"
			current_scale = random.choice(scales)

		new_note = ["note", t * duration, duration - 0.05, channel, pitch, loudness]
		# print(new_note)
		notes.append(new_note)
	for note in notes:
		note[1] = transform_time(note[1], 0)
		note[2] = transform_duration(note[2])

	return notes


def get_beats(n_measures, n_beats, tempo_factor):
	result = []
	for m in range(n_measures * n_beats):
		t = m * tempo_factor
		duration = 1 * tempo_factor
		pitch_encoded_instrument = 35 if m % n_beats == 0 else 37
		loudness = 60
		note = ["note", t, duration, 9, pitch_encoded_instrument, loudness]
		result.append(note)
	return result


def bpm_to_factor(tempo):
	return 6000 / tempo


def print_note_times(notes, tempo_factor, n_beats):
	print(tempo_factor)
	last_t = -1
	for note in notes:
		t = note[1]
		t_adjusted = ((t / tempo_factor) % n_beats)
		if t_adjusted > n_beats - EPSILON:
			t_adjusted = 0.0

		if t_adjusted <= last_t:
			print()
		print("%.3f" % t, "%.3f" % t_adjusted, end=", ")
		last_t = t_adjusted
	print()


scales = [
	structure.SCALES.mayamalavagowla,
	structure.SCALES.phrygian,
	structure.SCALES.phrygian_dominant,
	structure.SCALES.wangandar,
	structure.SCALES.harmonic_minor,
	structure.SCALES.octatonic_major,
	structure.SCALES.octatonic_minor,
]

# notes = get_notes(1000, scale=random.choice(scales), tempo_factor=random.randint(60, 120))
tempo = 130 # random.randint(80, 150)
print(tempo)
tempo_factor = bpm_to_factor(tempo)
resolution = 1
n_measures = 100
n_beats = 4

# notes = get_notes_with_measures(n_measures, n_beats, resolution, random.choice(scales), tempo_factor)
notes = get_colorful_walk(10000, scales, tempo_factor)
# beats = get_beats(n_measures, n_beats, tempo_factor)
beats = []

instrument = 1 - 1

if len(beats) > 0:
	my_score = [
	    100,
	    [   # track 0:
	        ['patch_change', 0, 1, instrument],
	    ] + notes + [
	    ],
	    [	# percussion track
			['patch_change', 0, 9, 0],
		] + beats + [
	    ]
	]
else:
	my_score = [
	    100,
	    [   # track 0:
	        ['patch_change', 0, 1, instrument],
	    ] + notes + [
	    ]
	]


with open("Music\Composition.mid", "wb") as f:
	f.write(midi.score2midi(my_score))