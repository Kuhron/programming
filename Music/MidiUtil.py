from datetime import datetime
import os
import pickle
import math
import numpy as np
import random
import time

import pygame
import pygame.midi as midi
midi.init()

import Music.MusicalStructureUtil as structure


class MidiEvent:
    def __init__(self, status, pitch, event, data3, timestamp):
        self.status = status
        self.pitch = pitch
        self.event = event
        self.data3 = data3
        self.timestamp = timestamp
        self.loudness = 64  # max is 127

        self.status_name = "note" if status == 144 else "instrument" if status == 192 else "unknown_status"
        if self.status_name == "note":
            self.event_name = "note_on" if self.event == 75 else "note_off" if self.event == 0 else "unknown_event"
        else:
            self.event_name = "N/A"

    def to_raw_data(self):
        return [[self.status, self.pitch, self.event, self.data3], self.timestamp]

    @staticmethod
    def from_raw_data(data):
        lst, timestamp = data
        args = lst + [timestamp]
        return MidiEvent(*args)

    @staticmethod
    def from_data_list(lst):
        return [MidiEvent.from_raw_data(x) for x in lst]

    def is_note(self):
        return self.status_name == "note"

    def is_instrument(self):
        return self.status_name == "instrument"

    def invert_pitch(self, pivot):
        if self.is_note():
            new_pitch = pivot + (pivot - self.pitch)
        return self

    def add_time(self, time):
        self.timestamp += time
        return self

    # def send_to_standard_out(self, player):
    #     if self.is_note():
    #         if self.event_name == "note_on":
    #             player.note_on(self.pitch, self.loudness)
    #         elif self.event_name == "note_off":
    #             player.note_off(self.pitch, self.loudness)
    #         else:
    #             raise Exception("cannot send event: {}".format(self.to_raw_data()))
    #     elif self.is_instrument():
    #         player.set_instrument(self.pitch)

    def __repr__(self):
        return str(self.to_raw_data())


def get_input_and_output_devices(verbose=False):
    INTERFACE_NAME = b"UM-2"  # Edirol UM-2 EX
    # INTERFACE_OTHER_NAME = b"MIDIOUT2 (UM-2)"
    # INTERFACE_NAME = b"US-144"  # Tascam US-144

    infos = [midi.get_device_info(device_id) for device_id in range(midi.get_count())]
    if verbose:
        print("got midi infos:", infos)

    input_device_id = None
    output_device_id = None
    alt_device_id = None
    for device_id, info in enumerate(infos):
        interf, name, is_input, is_output, is_opened = info
        if name == INTERFACE_NAME:
            if is_input:
                input_device_id = device_id
            elif is_output:
                output_device_id = device_id
        elif name == INTERFACE_OTHER_NAME:
            raise Exception("OTHER_NAME should not be used; check that input/output device names are as you expect")
            alt_device_id = device_id

    if input_device_id is None or output_device_id is None:
        print("MIDI input and/or output could not be correctly identified. Result: input {}; output {}".format(input_device_id, output_device_id))
    inp = midi.Input(input_device_id) if input_device_id is not None else None
    outp = midi.Output(output_device_id, latency=1) if output_device_id is not None else None # if latency is 0 then timestamps are ignored by pygame

    return inp, outp


def send_data_to_midi_out(data, midi_output):
    pygame_time_ms = midi.time()
    transform = lambda lst, timestamp: [lst, timestamp + pygame_time_ms + 1000]
    data = [transform(*x) for x in data]

    final_timestamp = data[-1][-1]

    # kill time so program doesn't end before midi is done playing

    i = 0
    MAX_OUTPUT_LENGTH = 1024  # due to pypm
    while i < len(data):
        sub_data = data[i: i + MAX_OUTPUT_LENGTH]
        midi_output.write(sub_data)
        i += MAX_OUTPUT_LENGTH

    while midi.time() < final_timestamp:
        time.sleep(0.1)


def send_notes_to_midi_out(notes, midi_output):
    for note in notes:
        assert type(note) in [structure.Note, structure.Chord, structure.Rest], (
            "note must be Music\\MusicalStructureUtil.Note or Chord, not {}. object received: {}".format(type(note), note)
        )
        note.output_to_midi(midi_output)


def send_events_to_standard_out(events):
    player = midi.Output(0)
    data = [x.to_raw_data() for x in events]
    send_data_to_midi_out(data, player)
    return    

    # pygame_time_ms = midi.time()
    # transform = lambda event: event.add_time(pygame_time_ms + 1000)
    # events = [transform(x) for x in events]

    # final_timestamp = events[-1].timestamp

    # for event in events:
    #     print(event)
    #     event.send_to_standard_out(player)

    # # kill time so program doesn't end before midi is done playing

    # # i = 0
    # # MAX_OUTPUT_LENGTH = 1024  # due to pypm
    # # while i < len(data):
    # #     sub_data = data[i: i + MAX_OUTPUT_LENGTH]
    # #     midi_output.write(sub_data)
    # #     i += MAX_OUTPUT_LENGTH

    # while midi.time() < final_timestamp:
    #     time.sleep(0.1)


def read_data_from_midi_in(midi_input, max_silence_seconds):
    data = []
    t0 = time.time()
    last_time = None
    note_imbalance = 0
    while True:
        if midi_input.poll():
            new_data = midi_input.read(1)
            assert len(new_data) == 1
            new_data = new_data[0]
            data.append(new_data)
            print(new_data)
            event = MidiEvent.from_raw_data(new_data)
            last_time = event.timestamp
            
            note_imbalance += 1 if event.event_name == "note_on" else -1 if event.event_name == "note_off" else 0
            # ? = data3
            print("notes on:", note_imbalance)
        elif note_imbalance == 0 and last_time is not None and midi.time() - last_time > max_silence_seconds * 1000:
            print("data collection timed out")
            break
    return data


def read_notes_from_midi_in(midi_input, timeout_seconds):
    data = read_data_from_midi_in(midi_input, timeout_seconds)
    # TODO: parse data
    raise NotImplementedError


def dump_data(data):
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = "Music/midi_input_{}.pickle".format(now_str)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_random_data():
    data_dir = "Music/"
    ls = os.listdir(data_dir)
    choices = [x for x in filter(lambda x: x.startswith("midi_input_"), ls)]
    # print(choices)
    choice = random.choice(choices)
    with open(data_dir + choice, "rb") as f:
        data = pickle.load(f)
    return data


def load_data_from_datetime_string(s):
    filepath = "Music/midi_input_{}.pickle".format(s)
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def invert_data(data, pivot):
    lst = MidiEvent.from_data_list(data)
    lst = [x.invert_pitch(pivot) for x in lst]
    return [x.to_raw_data() for x in lst]


def note_number_to_hertz(n, a=440):
    # a440 is 69, c0 ~=16 Hz is 0
    deviation_from_a440 = (n - 69)/12
    a440_freq = a
    factor_deviation = 2 ** deviation_from_a440
    return a440_freq * factor_deviation


def hertz_to_note_number(hz, a=440):
    # a440 is 69, c0 ~=16 Hz is 0
    log2_hz = math.log(hz, 2)
    log2_a440 = math.log(a, 2)
    deviation_in_logs = log2_hz - log2_a440
    deviation_in_semitones = deviation_in_logs*12
    return 69 + deviation_in_semitones


def note_number_to_name(n):
    pitch_class = n % 12
    octave = (n // 12) - 1  # C-1 ~= 8 Hz is number 0; C0 ~= 16 Hz is number 12
    if pitch_class % 1 == 0:
        pitch_class = int(pitch_class)
        letter = "CKDHEFXGJARB"[pitch_class]
    else:
        letter = "?"
    assert octave % 1 == 0
    octave = int(octave)
    return letter + str(octave)


def test_note_number_math():
    assert hertz_to_note_number(440) == 69, "440 Hz is #{}, not #69".format(hertz_to_note_number(440))

    n = 0
    h = note_number_to_hertz(n)
    nn = hertz_to_note_number(h)
    assert abs(nn - n) < 1e-6, "n={}, h={}, nn={}".format(n, h, nn)

    n = 69 + 12
    h = note_number_to_hertz(n)
    assert abs(h - 880) < 1e-6, "n={}, h={}, should be 880".format(n, h)
    nn = hertz_to_note_number(880)
    assert abs(nn - n) < 1e-6, "nn from 880 = {}, should be {}".format(nn, n)

    n = -25
    h = note_number_to_hertz(n)
    nn = hertz_to_note_number(h)
    assert abs(nn - n) < 1e-6, "n={}, h={}, nn={}".format(n, h, nn)

    n = 106.1315126
    h = note_number_to_hertz(n)
    nn = hertz_to_note_number(h)
    assert abs(nn - n) < 1e-6, "n={}, h={}, nn={}".format(n, h, nn)

    n = np.pi * np.exp(np.pi)
    h = note_number_to_hertz(n)
    nn = hertz_to_note_number(h)
    assert abs(nn - n) < 1e-6, "n={}, h={}, nn={}".format(n, h, nn)


