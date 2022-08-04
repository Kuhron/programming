import sys
sys.path.insert(0,'..')

import Music.WavUtil as wav
import Music.MidiUtil as midi_util


for note_number in range(0, 109):
    fp = f"/home/wesley/linux-tone-keyboard/WavTones/{note_number}.wav"
    hz = midi_util.note_number_to_hertz(note_number, a=432)
    print(f"note number {note_number} is {hz} Hz")
    signal = wav.get_signal_from_freq(hz, seconds=0.2, truncate="forward")
    print(signal.shape)
    wav.write_signal_to_wav(signal, fp, amplitude_fraction=0.15)
