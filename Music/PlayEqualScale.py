import sys
sys.path.insert(0, "..")

import numpy as np
import Music.WavUtil as wav
import Music.MidiUtil as midi_util


print()
n = int(input("How many steps do you want to divide the octave into? "))
assert n >= 2, "need at least 2 steps"

f0 = midi_util.note_number_to_hertz(60, a=432)
full_signal = np.array([])
for i in range(0, n+1):
    r = 2 ** (i/n)
    f = f0 * r
    print(f"scale step {i}/{n}, pitch ratio {r:.4f}")
    signal = wav.get_signal_from_freq(f, seconds=0.2, truncate="forward", initial_click=True)
    full_signal = np.concatenate([full_signal, signal])

full_signal = wav.pad_signal_with_silence(full_signal, 0.5)
wav.send_signal_to_audio_out(full_signal)
