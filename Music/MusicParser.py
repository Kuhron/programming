import Music.MusicalStructureUtil as structure
import Music.WavUtil as wav


TEMPO = 150
WORD_GAP = structure.Rest(structure.Duration(0, TEMPO))
SENTENCE_GAP = structure.Rest(structure.Duration(1, TEMPO))
DEFAULT_OCTAVE = 4


def parse_note(note, last_note=None):
    assert type(note) is str and 0 < len(note), "Invalid input: {}".format(repr(note))
    assert note[0] in structure.PITCH_CLASSES, "invalid pitch class: {}".format(repr(note[0]))
    # assert note[1] in structure.OCTAVES, "invalid octave {}".format((note[1]))

    pitch_class = note[0]
    # octave = int(note[1])

    octave = get_octave_from_pitch_class_and_last_note(pitch_class, last_note)

    modifiers = note[1:]

    duration = structure.Duration(0.5, TEMPO)  # default

    for modifier in modifiers:
        pitch_class, octave, duration = modify(modifier, pitch_class, octave, duration)

    octave = cap_octave(octave)

    name = pitch_class + str(octave)

    return structure.Note(name, duration)


def modify(modifier, pitch_class, octave, duration):
    if modifier in structure.OCTAVES:
        octave = int(modifier)
    elif modifier == "+":
        octave += 1
    elif modifier == "-":
        octave -= 1
    else:
        raise ValueError("invalid modifier {}".format(modifier))

    return pitch_class, octave, duration


def cap_octave(octave):
    return max(structure.MIN_OCTAVE, min(structure.MAX_OCTAVE, octave))


def get_octave_from_pitch_class_and_last_note(pitch_class, last_note):
    if last_note is None:
        return DEFAULT_OCTAVE

    if type(last_note) is structure.Chord:
        return get_octave_from_pitch_class_and_last_note(pitch_class, last_note.notes[0])

    last_pitch_class = last_note.pitch_class
    last_octave = last_note.octave

    last_index = structure.PITCH_CLASSES.index(last_pitch_class)
    current_index = structure.PITCH_CLASSES.index(pitch_class)
    distance_up = (current_index - last_index) % 12
    distance_down = (last_index - current_index) % 12
    if distance_up == distance_down == 0:
        return last_octave
    elif distance_up == distance_down == 6:
        octave_direction = "up"
    elif distance_up == distance_down:
        raise Exception("invalid distances")
    elif distance_up < distance_down:
        octave_direction = "up"
    else:
        octave_direction = "down"

    if octave_direction == "up":
        if current_index > last_index:
            octave = last_octave
        else:
            octave = last_octave + 1
    else:
        if current_index < last_index:
            octave = last_octave
        else:
            octave = last_octave - 1

    return octave


def get_notes_from_cluster(word):
    notes = []
    current_note = ""
    for char in word:
        if char in structure.PITCH_CLASSES:
            if current_note != "":
                notes.append(current_note)
            current_note = ""
        current_note += char
    notes.append(current_note)
    return notes


def parse_notes_from_list(notes, last_note):
    res = []
    for note in notes:
        if note == "":
            continue
        parsed = parse_note(note, last_note)
        res.append(parsed)
        last_note = parsed
    return res, last_note


def parse_word(word, last_note=None):
    parsed = []

    split_left_paren = word.split("(")
    for segment in split_left_paren:
        if segment == "":
            continue

        if ")" in segment:
            in_chord, out_of_chord = segment.split(")")
        else:
            in_chord = ""
            out_of_chord = segment

        in_chord_notes = get_notes_from_cluster(in_chord)
        out_of_chord_notes = get_notes_from_cluster(out_of_chord)

        in_chord_parsed, last_note = parse_notes_from_list(in_chord_notes, last_note)
        out_of_chord_parsed, last_note = parse_notes_from_list(out_of_chord_notes, last_note)

        if in_chord != "":
            chord_names = [x.name for x in in_chord_parsed]
            chord_duration = in_chord_parsed[0].duration
            parsed.append(structure.Chord(chord_names, chord_duration))
        parsed.extend(out_of_chord_parsed)

    return parsed

    # notes = get_notes_from_cluster(word)

    # res = []
    # for note in notes:
    #     parsed = parse_note(note, last_note)
    #     res.append(parsed)
    #     last_note = parsed
    # return res


def parse_sentence(s):
    words = [x for x in s.split(" ") if x != ""]

    res = []
    last_note = None
    for word in words:
        parsed = parse_word(word, last_note)
        res.append(parsed + [WORD_GAP])
        last_note = parsed[-1]
    # res = [parse_word(w) + [WORD_GAP] for w in words]
    return [item for lst in res for item in lst]


def parse_text(text):
    # text = add_octaves_to_text(text)
    # print(text)

    # don't let last-note dependence persist across sentences; just ignore all previous information
    sentences = [x for x in text.split("|") if x.replace(" ", "") != ""]
    res = [parse_sentence(s) + [SENTENCE_GAP] for s in sentences]
    return [item for lst in res for item in lst]


def parse_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]

    new_text = "|".join(filter((lambda x: x[0] != "#"), lines))

    return parse_text(new_text)


# def add_octaves_to_text(text):
#     last_octave = DEFAULT_OCTAVE
#     last_note = None
#     res = ""
#     for char in text:
#         res += char
#         if char in structure.PITCH_CLASSES:
#             if last_note is None or last_note == char:
#                 last_note = char
#                 res += str(last_octave)
#             else:
#                 last_index = structure.PITCH_CLASSES.index(last_note)
#                 current_index = structure.PITCH_CLASSES.index(char)
#                 distance_up = (current_index - last_index) % 12
#                 distance_down = (last_index - current_index) % 12
#                 # print("distance from {0} to {1}: up = {2}, down = {3}".format(last_note, char, distance_up, distance_down))
#                 # input("x")
#                 octave_direction = "up" if distance_up <= distance_down else "down"  # prefer up when tied

#                 if octave_direction == "up":
#                     if current_index > last_index:
#                         octave = last_octave
#                     else:
#                         octave = last_octave + 1
#                 else:
#                     if current_index < last_index:
#                         octave = last_octave
#                     else:
#                         octave = last_octave - 1

#                 last_note = char
#                 last_octave = octave

#                 res += str(octave)

#     return res


if __name__ == "__main__":
    # res = parse_file("Music\\MusicParserTestInput.txt")
    res = parse_file("Music\\MusicParserTestInputAdvanced.txt")
    # print(res)
    signal = wav.get_signal_from_notes(res)
    # wav.send_signal_to_audio_out(signal)
    wav.write_signal_to_wav(signal, "Music\\MusicOutput.wav")
