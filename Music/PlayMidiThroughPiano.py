import sys
sys.path.insert(0, "/home/wesley/programming/")

import Music.MidiUtil as mu
import os
import random


if __name__ == "__main__":
    inp, outp = mu.get_input_and_output_devices()
    data_dir = "/home/wesley/programming/Music/midi_input/YamahaP125"
    data = mu.load_random_data(data_dir)
    # data = mu.load_data_from_datetime_string(data_dir, "20231001-192251")
    start_time_s = 0
    start_time_ms = int(1000 * start_time_s)
    data = [[x, t-start_time_ms] for x,t in data if t >= start_time_ms]

    # mess with it
    if random.random() < 0.5:
        data = mu.invert_data(data, pivot=60)
    offset = random.randint(-6, 6)
    print(f"{offset = }")
    data = mu.transpose_data(data, offset)

    mu.send_data_to_midi_out(data, outp)
