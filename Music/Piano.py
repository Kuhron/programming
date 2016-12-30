import msvcrt
import pyaudio
import numpy as np
import time
import pygame
import winsound

import Music.WavUtil as wav
import Music.MusicalStructureUtil as structure


def get_note_name_from_key(c):
	# increment octave between B and C, according to https://en.wikipedia.org/wiki/Scientific_pitch_notation
    d = {
        b"1":"A5", b"2":"M5", b"3":"B5", b"4":"C6", b"5":"K6", b"6":"D6", b"7":"H6", b"8":"E6", b"9":"F6", b"0":"X6", b"[":"G6", b"]":"L6",
        b"'":"A4", b",":"M4", b".":"B4", b"p":"C5", b"y":"K5", b"f":"D5", b"g":"H5", b"c":"E5", b"r":"F5", b"l":"X5", b"/":"G5", b"=":"L5",
        b"a":"A3", b"o":"M3", b"e":"B3", b"u":"C4", b"i":"K4", b"d":"D4", b"h":"H4", b"t":"E4", b"n":"F4", b"s":"X4", b"-":"G4", b"\r":"L4",
        b"\\":"A6"
    }
    return d.get(c)


def beep(sound="tone"):
    winsound.PlaySound('%s.wav' % sound, winsound.SND_FILENAME)


def play_old():
	pygame.mixer.init(wav.RATE,-16,2,4096)

	sound_file = "tone.wav"
	sound_length = 0.1

	sound = pygame.mixer.Sound(sound_file)
	snd_array = pygame.sndarray.array(sound)
	snd_out = pygame.sndarray.make_sound(snd_array)

	shift = int(input("Default home pitch: A4. Number of semitones to shift: "))

	while True:
		if msvcrt.kbhit():
			if True: #for c in msvcrt.getch():
				c = msvcrt.getch()
				# snd_out.play()
				freq = structure.get_frequency_from_note_name(get_note_name_from_key(c), shift)
				if freq:
					seconds = 0.05
					winsound.Beep(freq, int(seconds*1000))
					# beep()


def play():
	audio_out = pyaudio.PyAudio()
	stream = audio_out.open(
	    format=pyaudio.paInt8,
	    channels=1,
	    rate=wav.RATE,
	    output=True,
	)

	shift = int(input("Default home pitch: A4. Number of semitones to shift: "))

	current_freq = None
	time_pressed = None
	min_duration = 0.03
	max_duration = 0.5
	while True:
		now = time.time()
		if msvcrt.kbhit():
			c = msvcrt.getch()
			name = get_note_name_from_key(c)
			if name is None:
				continue
			freq = structure.get_frequency_from_note_name(name, shift)
			if freq:
				current_freq = freq
				time_pressed = now
		if current_freq is not None:
			if now > time_pressed + max_duration:
				current_freq = None
				time_pressed = None
			else:
				wav.send_freq_to_stream(current_freq, min_duration, stream, initial_click=True)

	stream.stop_stream()
	stream.close()
	audio_out.terminate()




if __name__ == "__main__":
	play()













