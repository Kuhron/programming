import os
import random

import MidiUtil as mu


PARENT_DIR = "/home/wesley/programming/Music/midi_input/YamahaP125/"
INP, OUTP = mu.get_input_and_output_devices()
print(f"{INP=}, {OUTP=}")
if OUTP is None:
    PLAY_FUNC = mu.send_data_to_standard_out
else:
    PLAY_FUNC = lambda data: mu.send_data_to_midi_out(data, OUTP)


class Piece:
    def __init__(self, fname_str, parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None):
        # parent_drumtrack is the drumtrack that was played while this piece was improvised, if any
        # parent_pieces are the pieces that were played while this piece was improvised, if any
        # parent_piece_transformations tell what was done to the parent pieces (e.g. inversion, transposition) before they were played for this piece to be improvised over

        self.fname_str = fname_str
        self.fname = f"midi_input_{self.fname_str}.pickle"
        self.fp = os.path.join(PARENT_DIR, self.fname)
        assert os.path.exists(self.fp)
        self.midi_data = mu.load_data_from_fname_string(PARENT_DIR, self.fname_str)

        if parent_pieces is None:
            parent_pieces = []
        if parent_piece_transformations is None:
            parent_piece_transformations = []
        assert len(parent_pieces) == len(parent_piece_transformations)

    def play_alone(self):
        print(f"playing {self.fname_str}")
        PLAY_FUNC(self.midi_data)
        print("done playing")

    def play_combined(self):
        print(f"playing {self.fname_str}")
        midi_data = self.midi_data
        for p, t in zip(self.parent_pieces, self.parent_piece_transformations):
            new_midi_data = t.transform(p.midi_data)
            midi_data += new_midi_data
        midi_data += self.parent_drumtrack.midi_data
        midi_data = sorted(midi_data, key=lambda x: x[-1])
        PLAY_FUNC(midi_data)
        print("done playing")


class Drumtrack:
    def __init__(self):
        # get the drumtrack midi event list
        raise NotImplementedError


class Transformation:
    def __init__(self, inverted, transposition, time_offset, time_dilation):
        self.inverted = inverted
        self.transposition = transposition
        self.time_offset = time_offset
        self.time_dilation = time_dilation

    @staticmethod
    def identical():
        return Transformation(inverted=False, transposition=0, time_offset=0, time_dilation=1)

    def transform(self, midi_data):
        raise NotImplementedError


pieces = [
    # Piece("20230930-212933", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20230930-213126", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20230930-214134", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20231001-005902", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20231001-192251", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20231001-193721", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20231001-194005", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20231001-200554"),  # TODO COcta parent clipped to correct length
    # Piece("20231002-020531", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20231014-221751"),  # TODO COcta parent clipped to correct length
    # Piece("20231014-222344"),  # TODO COcta parent transformed and clipped to correct length
    # Piece("20231014-232607"),  # TODO COcta parent clipped to correct length
    # Piece("20231109-065457", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20231109-071345", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
    # Piece("20231109-071722", parent_pieces=None, parent_drumtrack=None, parent_piece_transformations=None),
]
# TODO add ability for parent track to be a repeater
# TODO add ability for drumtrack to start a certain number of measures early before the parent (whether the parent repeats or not) starts

p = random.choice(pieces)
p.play_alone()

