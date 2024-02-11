import sys
sys.path.insert(0, "/home/wesley/programming/")

import Music.MidiUtil as mu


if __name__ == "__main__":
    inp, outp = mu.get_input_and_output_devices(verbose=True)
    if inp is None:
        raise Exception("no input device")
    else:
        data = mu.read_data_from_midi_in(inp, max_silence_seconds=5)
        mu.dump_data(data)

