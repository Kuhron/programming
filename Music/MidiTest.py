import MidiUtil as mu


if __name__ == "__main__":
    max_silence_seconds = 1

    try:
        inp, outp = mu.get_input_and_output_devices(verbose=True)

        if inp is not None:
            # notes = mu.read_notes_from_midi_in(inp, timeout_seconds)
            data = mu.read_data_from_midi_in(inp, max_silence_seconds)
            mu.dump_data(data)

        # data = mu.load_random_data()
        # data = mu.invert_data(data, 66)

        # datetime_str = "20170220-010435"
        # data = mu.load_data_from_fname_string(datetime_str)

        if outp is not None:
            mu.send_data_to_midi_out(data, outp)

        # events = mu.MidiEvent.from_data_list(data)
        # mu.send_events_to_standard_out(events)
    except:
        raise
    finally:
        # pass
        if inp is not None:
            inp.close()
        if outp is not None:
            outp.close()
