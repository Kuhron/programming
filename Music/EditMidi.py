from mido import Message, MidiFile, MidiTrack
from copy import deepcopy






if __name__ == "__main__":
    input_fp = "/home/wesley/Desktop/Construction/MusicComposition/Wesley's/Year 1/24. Dance of the Haunts (Complete)/Ensemble/24. Dance of the Haunts (Complete).mid"
    mid = MidiFile(input_fp)
    new_mid = MidiFile()
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)
        for msg in track:
            if msg.is_meta:
                # don't change it, it's a MetaMessage
                new_msg = msg
            else:
                new_msg = deepcopy(msg)
                if hasattr(new_msg, "note"):
                    new_note = new_msg.note + 4
                    new_msg.note = int(new_note) % 127

            print(f"\nold message:\n{msg}\nwith dict {msg.dict()}\nnew message:\n{new_msg}\nwith dict {new_msg.dict()}\n")
            input("check")
            new_track.append(new_msg)

        new_mid.save('new_song.mid')

