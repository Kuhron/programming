# make a visualization of the notes over time, like a music staff

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import MidiUtil as mu


def plot_note_presses(events):
    assert all(type(x) is mu.MidiEvent for x in events)
    # put a line segment for note duration, with a dot for onset
    note_presses = get_note_presses(events)
    lines = []
    point_xs = []
    point_ys = []
    pitches = set()
    times = set()
    for pitch, on_time, off_time in note_presses:
        pitches.add(pitch)
        times.add(on_time)
        times.add(off_time)
        p0 = [on_time, pitch]
        p1 = [off_time, pitch]
        line = [p0, p1]
        lines.append(line)
        point_xs.append(on_time)
        point_ys.append(pitch)
    lc = LineCollection(lines, colors="k")
    fig, ax = plt.subplots()

    # add staves, going to do my own way that makes more sense than traditional grand staff (especially because each semitone is its own y value here)
    # middle C gets a special noticeable color, every other C is also fairly noticeable but not as prominent
    # other notes are either left blank or color-coded and the same in each octave
    pitch_to_color = {}
    goldenrod = "#daa520"
    yellow = "#ffff00"
    magenta = "#ff00ff"
    beige = "#d9b99b"
    blue = "#0000ff"
    red = "#ff0000"
    green = "#00ff00"

    pitch_to_color[60] = goldenrod
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    min_time = min(times)
    max_time = max(times)

    cs = [-24, -12, 0, 12, 24, 36, 48, 60, 72, 84, 108, 120, 132, 144]
    for pitch in cs:
        if min_pitch <= pitch <= max_pitch and pitch != 60:
            pitch_to_color[pitch] = yellow
    offsets = [-4, -2, 2, 4, 6]
    for offset in offsets:
        color = {-4: magenta, -2: beige, 2: blue, 4: red, 6: green}[offset]
        for pitch in cs:
            pitch += offset
            if min_pitch <= pitch <= max_pitch:
                pitch_to_color[pitch] = color

    staff_lines = []
    staff_line_colors = []
    print(sorted(pitch_to_color.keys()))
    for pitch, color in pitch_to_color.items():
        p0 = [min_time, pitch]
        p1 = [max_time, pitch]
        line = [p0, p1]
        staff_lines.append(line)
        staff_line_colors.append(color)

    lc2 = LineCollection(staff_lines, colors=staff_line_colors)
    ax.add_collection(lc2)

    # now add notes on top of the staff lines
    ax.add_collection(lc)
    ax.scatter(point_xs, point_ys, c="k")
    ax.autoscale()
    ax.margins(0.1)

    plt.show()


def get_note_presses(events):
    # get list of notes with pitch, on time, off time
    on_time_by_pitch = {}
    notes = []
    for event in events:
        event_name = event.event_name
        pitch = event.pitch
        if event.timestamp == 0:
            raise Warning("event.timestamp is exactly zero, likely mido did not record the time and it was not added later in my own code")
        if event_name == "note_on":
            on_time = on_time_by_pitch.get(pitch)
            if on_time is not None:
                raise Exception(f"note {pitch} is already on! {event = }")
            on_time_by_pitch[pitch] = event.timestamp
        elif event_name == "note_off":
            on_time = on_time_by_pitch.get(pitch)
            if on_time is None:
                raise Exception(f"note {pitch} is not on yet! {event = }")
            off_time = event.timestamp
            on_time_by_pitch[pitch] = None
            note = [pitch, on_time, off_time]
            notes.append(note)
        else:
            # ignore it
            pass

    # any remaining on notes need to be turned off
    for pitch, on_time in on_time_by_pitch.items():
        if on_time is not None:
            off_time = events[-1].timestamp
            note = [pitch, on_time, off_time]
            notes.append(note)
    return notes

if __name__ == "__main__":
    
    data_dir = "/home/wesley/programming/Music/DigitalPiano/midi_input/YamahaP125"
    data = mu.load_data_from_fname_string(data_dir, "20231002-020531", "txt")

    events = [mu.MidiEvent.from_raw_data(lst) for lst in data]
    plot_note_presses(events)
