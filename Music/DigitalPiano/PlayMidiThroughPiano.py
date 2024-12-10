import os
import random
import mido
import sys

sys.path.insert(0, "/home/wesley/programming/")
import Music.MidiUtil as mu


# def play_accompaniment(dirpath, fname, with_drumtrack=True):

# def get_accompaniment_midi(?):

# def get_drumtrack_midi(?):

# def combine_midis(?):

# for a given dir and fname, get the midi for the piano accompaniment without drumtrack, and separately get the drumtrack
# then can have the same drumtrack all the way down the duet tree
# and for making a child of the root, just treat the original accompaniment notes (without drumtrack) the same as you would a later improvisation child


if __name__ == "__main__":
    try:
        s = sys.argv[1]
    except IndexError:
        s = None

    # inp, outp = mu.get_input_and_output_devices()
    inp, outp = mu.get_digital_piano_input_and_output()
    data_dir = "/home/wesley/programming/Music/DigitalPiano/midi_input/YamahaP125"
    # mu.verify_data_list_format_for_files_in_dir(data_dir)

    # data = mu.load_random_data(data_dir)
    # data = mu.load_data_from_fname_string(data_dir, "20231109-065457")
    # data = mu.load_data_from_fname_string(data_dir, "20231014-222344")
    # data = mu.load_data_from_fname_string(data_dir, "20231002-020531", "txt")  # name this one "Land of Ash", "Rivers of Ash" or something similar
    data = mu.load_data_from_fname_string(data_dir, "20241210-215609", "txt")
    # good one: 20231002-020531, either inverted or not; inverted +5 gives nice B/F# key in the second half
    # data = mu.load_data_from_fname_string(data_dir, "20231130-074300")

    if s is not None:
        data = mu.load_data_from_fname_string(data_dir, s)

    # 20240201-000329 was recorded at the same time as playing 20240131-235023, 20240201-000329 is supposed to be piano but the program doesn't seem to know that
    # 20240222-070311 was played while 20240222-071155 was recorded

    nwc_parent_dir = "/home/wesley/Desktop/Construction/MusicComposition/Wesley's/Piano Accompaniments/"
    accompaniment_fstr = "COcta_20231001-200554"
    accompaniment_fname = f"{accompaniment_fstr}_accompaniment.mid"
    drumtrack_fname = f"{accompaniment_fstr}_drumtrack.mid"
    accompaniment_fp = os.path.join(nwc_parent_dir, accompaniment_fname)
    drumtrack_fp = os.path.join(nwc_parent_dir, drumtrack_fname)

    # mu.send_midi_file_to_port(accompaniment_fp, outp)
    # sys.exit()

    # accompaniment_midi = mido.MidiFile(accompaniment_fp)
    # for msg in accompaniment_midi.play():
    #     print(msg)
    #     input("a")
    # raise

    # TODO use mido library to read the events from the drumtrack file, shift the times of the events in `data` such that its first note matches the first note of the third measure of the drumtrack: https://www.twilio.com/blog/working-with-midi-data-in-python-using-mido

    # shift the times in the file if necessary
    start_time_s = 0
    dilation = 1
    start_time_ms = int(1000 * start_time_s)
    transform_time = lambda t: dilation * (t - start_time_ms)
    data = [[x, transform_time(t)] for x,t in data if t >= start_time_ms]

    # mess with it
    invert, offset = True, 0
    # invert, offset = True, 5
    # invert = random.random() < 0.5
    # offset = random.randint(-6, 6)

    if invert:
        data = mu.invert_data(data, pivot=60)
    print(f"{offset = }")
    data = mu.transpose_data(data, offset)

    mu.send_data_to_midi_out(data, outp)
    # mu.send_data_to_standard_out(data)
