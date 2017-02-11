import time

import pygame
import pygame.midi as midi
midi.init()

import Music.MusicalStructureUtil as structure


def get_input_and_output_devices():
    # Casio LK-43
    CASIO_KEYBOARD_NAME = b"UM-2"  # two inputs, each with this name, 
    CASIO_KEYBOARD_OTHER_NAME = b"MIDIOUT2 (UM-2)"

    infos = [midi.get_device_info(device_id) for device_id in range(midi.get_count())]
    # print(infos)

    input_device_id = None
    output_device_id = None
    alt_device_id = None
    for device_id, info in enumerate(infos):
        interf, name, is_input, is_output, is_opened = info
        if name == CASIO_KEYBOARD_NAME:
            if is_input:
                input_device_id = device_id
            elif is_output:
                output_device_id = device_id
        elif name == CASIO_KEYBOARD_OTHER_NAME:
            alt_device_id = device_id

    # print(input_device_id, output_device_id)

    inp = midi.Input(input_device_id)
    outp = midi.Output(output_device_id)

    return inp, outp

    print(inp, outp)


def send_notes_to_midi_out(notes, midi_output):
    for note in notes:
        assert type(note) in [structure.Note, structure.Chord, structure.Rest], (
            "note must be Music\\MusicalStructureUtil.Note or Chord, not {}. object received: {}".format(type(note), note)
        )
        note.output_to_midi(midi_output)


