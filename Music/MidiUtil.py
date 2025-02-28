from datetime import datetime
import os
import pickle
import math
import numpy as np
import random
import time

# import pygame
# pygame.init()  # for initializing pygame.midi.mixer, if I use that
# import pygame.midi as midi
# midi.init()
import mido

import sys
sys.path.insert(0, "/home/wesley/programming/")

# import Music.MusicalStructureUtil as structure

MIDI_INPUT_DIR = "/home/wesley/programming/Music/midi_input/"
TIMIDITY_PORT = 2


class MidiEvent:
    STATUS_CODE_TO_NAME = {
        144: "note_on",
        128: "note_off",
        176: "pedal",
        192: "instrument",  # Yamaha P125 does a lot of events for changing instrument
        999: "program_change",  # made-up value for this status name
        998: "unknown_control_change",  # made-up value for this status name
        997: "unknown_control_change_channel_1",
        899: "program_change_channel_1",
    }
    STATUS_NAME_TO_CODE = {v:k for k,v in STATUS_CODE_TO_NAME.items()}

    def __init__(self, status, pitch, event, data3, timestamp):
        self.status = status
        self.pitch = pitch
        self.event = event
        assert data3 == 0, f"non-zero data3: {data3!r}"
        self.data3 = data3
        self.timestamp = timestamp
        self.loudness = 64  # max is 127

        self.status_name = MidiEvent.STATUS_CODE_TO_NAME.get(status, "unknown status")

        if self.status_name == "note_on":
            pressure = event
            self.loudness = pressure
            self.event_name = "note_on"
        elif self.status_name == "note_off":
            assert event == 64
            self.event_name = "note_off"
        elif self.status_name == "pedal":
            if event == 127:
                self.event_name = "pedal_on"
            elif event == 0:
                self.event_name = "pedal_off"
            else:
                self.event_name = "unknown pedal event"
        elif self.status_name == "instrument":
            self.event_name = "instrument"
        elif self.status_name == "program_change":
            self.event_name = "program change"
        elif self.status_name.startswith("program_change_channel_"):
            self.event_name = self.status_name
        elif self.status_name == "unknown_control_change":
            self.event_name = "unknown control change"
        elif self.status_name.startswith("unknown_control_change_channel_"):
            self.event_name = self.status_name
        else:
            self.event_name = "unknown event"

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

    def to_mido_message(self):
        if self.event_name in ["note_on", "note_off"]:
            msg_type = self.event_name
            msg_kwargs = {
                "channel": 0,
                "note": self.pitch,
                "velocity": self.loudness,
            }
        elif self.event_name in ["pedal_on", "pedal_off"]:
            msg_type = "control_change"
            msg_kwargs = {
                "channel": 0,
                "control": 64,
                "value": 127 if self.event_name == "pedal_on" else 0,
            }
        elif self.event_name == "instrument":
            msg_type = "sysex"
            assert type(self.event) is tuple and all(type(x) is int for x in self.event)
            msg_kwargs = {
                "data": self.event,
            }
        elif self.event_name == "program change":
            msg_type = "program_change"
            msg_kwargs = {
                "channel": 0,
                "program": self.pitch,
            }
        elif self.event_name.startswith("program_change_channel_"):
            channel_number = int(self.event_name.replace("program_change_channel_", ""))
            msg_type = "program_change"
            msg_kwargs = {
                "channel": channel_number,
                "program": self.pitch,
            }
        elif self.event_name == "unknown control change":
            msg_type = "control_change"
            msg_kwargs = {
                "channel": 0,
                "control": self.pitch,
                "value": self.event,
            }
        elif self.event_name.startswith("unknown_control_change_channel_"):
            channel_number = int(self.event_name.replace("unknown_control_change_channel_", ""))
            msg_type = "control_change"
            msg_kwargs = {
                "channel": channel_number,
                "control": self.pitch,
                "value": self.event,
            }
        else:
            print(f"ignoring event {self}")
            input("check, is this supposed to happen?\n^^^^\n")
            return None
            # raise Exception(self)
        msg_kwargs.update({"time": self.timestamp/1000})
        return mido.Message(msg_type, **msg_kwargs)

    def is_note(self):
        return self.status_name in ["note_on", "note_off"]

    def is_instrument(self):
        return self.status_name == "instrument"

    def invert_pitch(self, pivot):
        if self.is_note():
            new_pitch = pivot + (pivot - self.pitch)
        else:
            new_pitch = self.pitch
        self.pitch = new_pitch
        return self

    def add_time(self, time):
        self.timestamp += time
        return self

    def __repr__(self):
        return f"<MidiEvent status={self.status} pitch={self.pitch} event={self.event} data3={self.data3} timestamp={self.timestamp} status_name={self.status_name} event_name={self.event_name}>"


def mido_message_to_data_list(msg):
    print(f"converting to data list: {msg = }")
    # need timestamp passed since mido doesn't give it to us
    if msg.time == 0:
        raise Warning("msg.time is exactly zero, which likely means the time was not recorded because mido does not receive it from the piano; make sure you assign msg.time as you receive messages")
    loudness = 64  # default if it's never overwritten
    if msg.type == "note_on":
        pitch = msg.note
        loudness = msg.velocity
        event_code = loudness  # a.k.a. pressure
        # event_name = "note_on"
        status_name = "note_on"
    elif msg.type == "note_off":
        pitch = msg.note
        loudness = msg.velocity
        assert loudness == 64, loudness
        event_code = loudness
        # event_name = "note_off"
        status_name = "note_off"
    elif msg.type == "control_change":
        if msg.control == 64:
            # pedal presses/releases
            pitch = 0
            if msg.value == 127:
                # event_name = "pedal_on"
                event_code = 127
                status_name = "pedal"
            elif msg.value == 0:
                # event_name = "pedal_off"
                event_code = 0
                status_name = "pedal"
            else:
                raise ValueError(f"unknown {msg.value = }")
        else:
            channel = msg.channel
            pitch = msg.control
            status_name = "unknown_control_change" if channel == 0 else f"unknown_control_change_channel_{channel}"
            event_code = msg.value
    elif msg.type == "sysex":
        # instrument changes
        # this one is more complicated because it just has an attribute "data" which is a tuple of numbers, length not guaranteed to be the same across instances
        # best to just store it as a string and re-parse to a tuple of ints when loading txt file
        pitch = 0
        status_name = "instrument"
        event_code = ".".join(str(x) for x in msg.data)
    elif msg.type == "program_change":
        # I don't know what this is, just record the data
        channel = msg.channel
        program = msg.program
        status_name = "program_change" if channel == 0 else f"program_change_channel_{channel}"
        pitch = program
        event_code = 0
    else:
        print(f"unknown {msg.type = } in {msg = }; skipping")
        return None
        # raise ValueError(f"unknown {msg.type = }")

    status_code = MidiEvent.STATUS_NAME_TO_CODE[status_name]
    data3 = 0
    timestamp = int(round(msg.time * 1000))
    lst = [[status_code, pitch, event_code, data3], timestamp]

    print(f"done converting to data list: {msg = }\n- returning {lst}\n")
    return lst


def get_digital_piano_input_and_output():
    device_name = "Digital Piano:Digital Piano MIDI 1 20:0"  # Yamaha P125
    try:
        inport = mido.open_input(device_name)
        outport = mido.open_output(device_name)
        return inport, outport
    except OSError:
        return None, None


def get_input_and_output_devices(verbose=False):
    raise Exception("deprecated")
    INTERFACE_NAME = "Digital Piano:Digital Piano MIDI 1 20:0"  # Yamaha P125 on mido
    # INTERFACE_NAME = b"Digital Piano MIDI 1"  # Yamaha P125 on pygame
    # INTERFACE_NAME = b"UM-2"  # Edirol UM-2 EX
    # INTERFACE_NAME = b"US-144"  # Tascam US-144

    # INTERFACE_OTHER_NAME = None
    # INTERFACE_OTHER_NAME = b"MIDIOUT2 (UM-2)"

    print(f"{midi.get_count()} devices found")
    infos = [midi.get_device_info(device_id) for device_id in range(midi.get_count())]
    if verbose:
        print("got midi infos:", infos)

    input_device_id = None
    output_device_id = None
    alt_device_id = None
    for device_id, info in enumerate(infos):
        interf, name, is_input, is_output, is_opened = info
        if name == INTERFACE_NAME and INTERFACE_NAME is not None:
            if is_input:
                input_device_id = device_id
            elif is_output:
                output_device_id = device_id
        elif name == INTERFACE_OTHER_NAME and INTERFACE_OTHER_NAME is not None:
            raise Exception("OTHER_NAME should not be used; check that input/output device names are as you expect")
            alt_device_id = device_id

    if input_device_id is None or output_device_id is None:
        print("MIDI input and/or output could not be correctly identified. Result: input {}; output {}".format(input_device_id, output_device_id))
    inp = midi.Input(input_device_id) if input_device_id is not None else None
    outp = midi.Output(output_device_id, latency=1) if output_device_id is not None else None # if latency is 0 then timestamps are ignored by pygame

    return inp, outp


def send_data_to_midi_out(data, midi_output):
    print(f"sending MIDI data to {midi_output=}")
    t0 = time.time()
    transform = lambda lst, timestamp: [lst, timestamp + 1000]
    data = [transform(*x) for x in data]

    final_timestamp = data[-1][-1]
    last_n_seconds_to_write = int(final_timestamp // 1000)

    if False:  # not needed now that I'm doing the time of sending myself
        # maybe break the data into 1-second chunks or some other buffering method so we don't get weird gaps when the piano is trying to process the commands
        MAX_OUTPUT_LENGTH = 1024  # due to pypm
        data_segments_by_second = []
        for second_i in range(last_n_seconds_to_write + 1):
            segment = [x for x in data if x[-1] // 1000 == second_i]
            if len(segment) > MAX_OUTPUT_LENGTH:
                raise Exception("too many events in one second")
                # hopefully shouldn't happen with any pieces I'm playing
            msgs = [raw_data_to_mido_message(lst) for lst in segment]
            data_segments_by_second.append(msgs)

        assert sum(len(x) for x in data_segments_by_second) == len(data), "wrong number of events in segments"

    msgs = [raw_data_to_mido_message(lst) for lst in data]
    t0 = time.time()

    pitch_is_on = {}
    try:
        for msg in msgs:
            assert type(msg) is mido.Message
            a = 0
            while time.time() - t0 < msg.time:
                time.sleep(0.001)
                a += 1
            # print(f"{a:4d} iterations in while loop")
            print(msg)
            midi_output.send(msg)

            # account for which notes are still pressed
            if msg.type == "note_on":
                pitch_is_on[msg.note] = True
            elif msg.type == "note_off":
                pitch_is_on[msg.note] = False
    except KeyboardInterrupt:
        # close the open notes
        for pitch, is_on in pitch_is_on.items():
            if is_on:
                msg = mido.Message("note_off", channel=0, note=pitch, velocity=64, time=0)
                midi_output.send(msg)
                print(f"turned off note {pitch}")
        raise
    return

    # old stuff below; as of 2024-04-27, Yamaha piano isn't paying attention to `time` attribute of mido messages and just does them upon receipt, so I'm doing the time buffering myself now
    # go ahead and write the first second, and always send the next second's data to the piano ahead of time
    for msg in data_segments_by_second[0]:
        midi_output.send(msg)
    last_n_seconds_written = 0
    while True:
        n_seconds_now = int(time.time() - t0)
        n_seconds_to_write = n_seconds_now + 1
        if last_n_seconds_written is None or n_seconds_to_write != last_n_seconds_written:
            segment = data_segments_by_second[n_seconds_to_write]

            for msg in segment:
                while time.time() - t0 < msg.time:
                    time.sleep(0.0001)
                print(msg)
                midi_output.send(msg)
                # fun to think about what small amounts of time inaccuracy will accumulate over a duet ancestry line due to milliseconds getting added/subtracted in code like this
            last_n_seconds_written = n_seconds_to_write
            print(f"wrote data for second {n_seconds_to_write} out of {last_n_seconds_to_write}", end="\n")
        if n_seconds_to_write == last_n_seconds_to_write:
            break
        time.sleep(0.1)

    wait_for_final_timestamp(final_timestamp, (lambda: time.time()))


def wait_for_final_timestamp(final_timestamp, get_time_func):
    # kill time so program doesn't end before midi is done playing
    while True:
        t = get_time_func()
        if t < final_timestamp:
            print(f"not done yet; {t = }, {final_timestamp = }")
            time.sleep(1)


def send_notes_to_midi_out(notes, midi_output):
    for note in notes:
        assert type(note) in [structure.Note, structure.Chord, structure.Rest], (
            "note must be Music\\MusicalStructureUtil.Note or Chord, not {}. object received: {}".format(type(note), note)
        )
        note.output_to_midi(midi_output)


def send_midi_file_to_standard_out(fp):
    port = get_standard_out_port()
    send_midi_file_to_port(fp, port)


def send_midi_file_to_port(fp, port):
    mid = mido.MidiFile(fp)
    for msg in mid.play():
        port.send(msg)


def get_standard_out_port():
    return mido.open_output("TiMidity:TiMidity port 0 128:0")


def send_data_to_standard_out(data):
    # player = midi.Output(TIMIDITY_PORT)  # doesn't work well with timidity, it wants to play the whole segment's notes all at once
    # send_data_to_midi_out(data, player)

    # pygame.mixer.music.load(data)  # needs a file, not just a list of data
    # pygame.mixer.music.play()

    msgs = []
    nums_to_print = []
    for lst in data:
        msg = raw_data_to_mido_message(lst)
        if msg is not None:
            msgs.append(msg)
            nums, t = lst
            nums_to_print.append(nums + [t])
    assert all(msgs[i].time <= msgs[i+1].time for i in range(len(msgs)-1)), "msgs out of order"

    final_timestamp = data[-1][-1]
    with get_standard_out_port() as out_port:
        print(f"{out_port = }")
        t0 = time.time()
        start_delay = 0.5
        for msg, nums in zip(msgs, nums_to_print):
            t = msg.time
            loops_wasted = 0
            while True:
                dt = time.time() - t0
                if dt >= t:
                    out_port.send(msg)

                    print_message_numbers = True
                    if print_message_numbers:
                        print("\t".join(str(x) for x in nums) + "\t" + str(final_timestamp))
                        # TODO implement a printing style that looks kind of like the piano keys being pressed (like the roller on a player piano)
                    else:
                        print(int(1000*msg.time), final_timestamp, end="\r")
                    # print(f"{loops_wasted = }")
                    break
                else:
                    loops_wasted += 1
                    time.sleep(0.001)
        # wait_for_final_timestamp(final_timestamp, (lambda: 1000*(time.time() - t0)))
    return


def send_events_to_standard_out(events):
    data = [x.to_raw_data() for x in events]
    send_data_to_standard_out(data)
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
        msg = midi_input.poll()
        if msg is not None:
            assert type(msg) is mido.Message
            t = time.time() - t0
            msg.time = t  # have to do this myself since mido doesn't get it from the piano
            data.append(msg)
            print(f"{msg = }")
            # event = MidiEvent.from_raw_data(new_data)
            # print(f"{event = }")
            # print(event.event_name)
            # last_time = event.timestamp
            last_time = msg.time
            
            note_imbalance += 1 if msg.type == "note_on" else -1 if msg.type == "note_off" else 0
            # ? = data3
            print("notes on:", note_imbalance)
        elif note_imbalance == 0 and last_time is not None and time.time() - last_time - t0 > max_silence_seconds:
            print("data collection timed out")
            break
    return data


def read_notes_from_midi_in(midi_input, timeout_seconds):
    data = read_data_from_midi_in(midi_input, timeout_seconds)
    # TODO: parse data
    raise NotImplementedError


def dump_data(data):
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = "/home/wesley/programming/Music/DigitalPiano/midi_input/YamahaP125/midi_input_{}.txt".format(now_str)
    with open(filepath, "w") as f:
        for msg in data:
            lst = mido_message_to_data_list(msg)
            if lst is not None:
                (a, b, c, d), t = lst
                s = "\t".join([str(x) for x in [a, b, c, d, t]])
                f.write(s + "\n")

    return {
        "fp": filepath,
        "now_str": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }

    # old, using pickle
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_random_data(data_dir):
    if data_dir is None:
        data_dir = "/home/wesley/programming/Music/midi_input"
    ls = os.listdir(data_dir)
    choices = [x for x in filter(lambda x: x.startswith("midi_input_"), ls)]
    choice = random.choice(choices)
    fp = os.path.join(data_dir, choice)
    return load_data_from_filepath(fp)


def load_data_from_fname_string(data_dir, s, extension="pickle"):
    fp = os.path.join(data_dir, f"midi_input_{s}.{extension}")
    return load_data_from_filepath(fp)


def load_data_from_filepath(fp):
    if fp.endswith(".pickle"):
        return load_pickle_data(fp)
    elif fp.endswith(".txt"):
        return load_text_data(fp)
    else:
        raise Exception(f"bad extension: {os.path.splitext(fp)[-1]}")


def load_pickle_data(fp):
    print(f"loading pickled midi data from {fp}")
    with open(fp, "rb") as f:
        data = pickle.load(f)
    return data


def load_text_data(fp):
    with open(fp) as f:
        lines = f.readlines()
    data = []
    for l in lines:
        s = l.strip().split("\t")
        status_code_str, pitch_str, event_code_str, data3_str, timestamp_str = s
        assert data3_str == "0", f"got non-zero data3: {data3_str!r}"

        if "." in event_code_str:
            # sysex message for instrument changes on Yamaha P125
            event_code = tuple(int(x) for x in event_code_str.split("."))
        else:
            event_code = int(event_code_str)

        status_code = int(status_code_str)
        pitch = int(pitch_str)
        data3 = int(data3_str)
        timestamp = int(timestamp_str)
        lst = [[status_code, pitch, event_code, data3], timestamp]
        data.append(lst)
    return data


def verify_data_list_format_for_filepath(fp):
    data = load_data_from_filepath(fp)
    assert type(data) is list
    for x in data:
        assert type(x) is list
        lst, t = x
        assert type(lst) is list
        assert all(type(y) is int for y in lst)
        assert type(t) is int


def verify_data_list_format_for_files_in_dir(d):
    for fname in os.listdir(d):
        fp = os.path.join(d, fname)
        verify_data_list_format_for_filepath(fp)


def raw_data_to_mido_message(lst):
    # maybe later can cut out the middleman of MidiEvent object, but fine for now
    event = MidiEvent.from_raw_data(lst)
    msg = event.to_mido_message()
    return msg


def invert_data(data, pivot):
    print("inverting data")
    lst = MidiEvent.from_data_list(data)
    lst = [x.invert_pitch(pivot) for x in lst]
    data = [x.to_raw_data() for x in lst]
    print("got inverted data")
    return data


def transpose_data(data, offset):
    print("transposing data")
    assert type(offset) is int
    if offset == 0:
        return data
    lst = MidiEvent.from_data_list(data)
    new_lst = []
    for x in lst:
        if x.is_note():
            x.pitch += offset
        new_lst.append(x)
    data = [x.to_raw_data() for x in new_lst]
    print("got transposed data")
    return data


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



