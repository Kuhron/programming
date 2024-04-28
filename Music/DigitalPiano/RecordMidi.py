import sys
sys.path.insert(0, "/home/wesley/programming/")

import Music.MidiUtil as mu


if __name__ == "__main__":
    inp, outp = mu.get_digital_piano_input_and_output()
    if inp is None:
        raise Exception("no input device")
    else:
        data = mu.read_data_from_midi_in(inp, max_silence_seconds=5)
        dump_res = mu.dump_data(data)

    playback = True
    if playback:
        print(f"\nplaying back what was recorded\n")
        fp = dump_res["fp"]
        data = mu.load_data_from_filepath(fp)
        mu.send_data_to_midi_out(data, outp)
        print("\nplayback complete\n")
