import random

import Music.MusicalStructureUtil as structure
import Music.MusicParser as parser
import Music.WavUtil as wav


def get_note():
    a = random.choice(structure.PITCH_CLASSES)
    if random.random() < 0.3:
        a += random.choice("+-")
    return a


def get_chord(n_notes):
    notes = []

    limiter = 0
    limit = 1000
    while limiter < limit and len(notes) < n_notes:
        limiter += 1
        new_note = get_note()
        if new_note not in notes:
            notes.append(new_note)
    if limiter >= limit:
        print("reached limit of {} tries when looking for note not in chord".format(limit))

    s = "(" + "".join(notes) + ")"
    return s


def get_note_or_chord():
    if random.random() < 0.5:
        return get_note()
    else:
        n = random.randint(2, 5)
        return get_chord(n)


def get_text():
    text = "| "
    for i in range(1000):
        text += (get_note_or_chord())
        if random.random() < 0.3:
            text += " "
    text += " |"
    return text


if __name__ == "__main__":
    text = get_text()
    with open("Music\\MusicOutput.txt", "w") as f:
        f.write(text)
    res = parser.parse_text(text)
    # print(res)
    signal = wav.get_signal_from_notes(res)
    # wav.send_signal_to_audio_out(signal)
    wav.write_signal_to_wav(signal, "Music\\MusicOutput.wav")