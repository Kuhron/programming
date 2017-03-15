import datetime
import pyaudio
import string
import time
import wave
import winsound

import numpy as np

import Music.MusicalStructureUtil as structure
import Music.MusicParser as parser
import Music.WavUtil as wav
import Terminal


TABLE = {
    "A": "(A3A4)",
    "B": "(B3B4)",
    "C": "(C4C5)",
    "D": "(D4D5)",
    "E": "(E4E5)",
    "F": "(F4F5)",
    "G": "(G4G5)",
    "H": "(H4H5)",
    "I": "(X4D5)",
    "J": "(K4A4)",
    "K": "(K4K5)",
    "L": "(L4L5)",
    "M": "(M3M4)",
    "N": "(H4M4)",
    "O": "(E4B4)",
    "P": "(G4H5)",
    "Q": "(F4K5)",
    "R": "(E4C5)",
    "S": "(D4M4)",
    "T": "(K4L4)",
    "U": "(B3G4)",
    "V": "(M3X4)",
    "W": "(L4E5)",
    "X": "(X4X5)",
    "Y": "(H4B4)",
    "Z": "(A4F5)",
    "0": "(C4M4)",
    "1": "(C4K4)",
    "2": "(C4D4)",
    "3": "(C4H4)",
    "4": "(C4E4)",
    "5": "(C4F4)",
    "6": "(C4X4)",
    "7": "(C4G4)",
    "8": "(C4L4)",
    "9": "(C4A4)",
    ".": "(G4D5G5)",
    "!": "(G4D5G5)",  # for simplicity
    "?": "(G4D5G5)",  # for simplicity
    ",": "(X4K5X5)",
    "\'": "(X4K5X5)",  # for simplicity
    " ": " | ",
}


class MusicTerminal(Terminal.Terminal):
    def __init__(self):
        super().__init__()

        self.update_bpm(120)

        self.initialize_commands()
        self.save_status = False
        self.initial_click = False
        self.play_status = True

    def initialize_commands(self):
        super().initialize_commands()
        
        self.add_command("bpm", self.process_bpm_input, "Change bpm to arg if given, else show current bpm.")
        self.add_command("s", self.change_save_status, "Turn on saving to file if arg is 1, else turn off if arg is 0.")
        self.add_command("p", self.change_play_status, "Turn on out-loud play if arg is 1, else turn off if arg is 0.")

    def change_save_status(self, a=None):
        return self.change_binary_attribute("save_status", a)

    def change_play_status(self, a=None):
        return self.change_binary_attribute("play_status", a)

    def process_normal_input(self, user_input):
        self.play(user_input)

    def process_bpm_input(self, bpm=None):
        if bpm is not None:
            self.update_bpm(int(bpm))
            return
        else:
            return self.bpm

    def update_bpm(self, bpm):
        self.bpm = int(bpm)

        self.duration = structure.Duration(1, self.bpm)

    def translate(self, s):
        s = [x for x in s.upper() if x in string.ascii_uppercase + "0123456789., "]
        text = [TABLE[x] for x in s]
        return "".join(text)

    def play(self, user_input):
        # audio_out = pyaudio.PyAudio()
        # rate = 44100
        # stream = audio_out.open(
        #     format=pyaudio.paInt8,
        #     channels=1,
        #     rate=rate,
        #     output=True,
        # )

        full_signal = []
        text = self.translate(user_input)

        res = parser.parse_text(text, self.bpm)
        signal = wav.get_signal_from_notes(res)
        wav.send_signal_to_audio_out(signal)

        # for x in morse:
        #     if x == " ":
        #         silence = wav.get_silence_for_duration((self.word_gap - self.letter_gap) / 1000)
        #         wav.send_signal_to_stream(silence, stream)
        #         if self.save_status:
        #             full_signal.extend(silence)
        #     else:
        #         durations = [self.dit if y == "." else self.dah for y in x]
        #         for duration in durations:
        #             # winsound.Beep(self.tone_hz, duration)
        #             signal = wav.get_signal_from_freq(self.tone_hz, duration / 1000, initial_click=self.initial_click)
        #             if self.play_status:
        #                 wav.send_signal_to_stream(signal, stream)
        #             if self.save_status:
        #                 full_signal.extend(signal)
        #             # self.sleep(self.beep_gap)
        #             silence = wav.get_silence_for_duration(self.beep_gap / 1000)
        #             if self.play_status:
        #                 wav.send_signal_to_stream(silence, stream)
        #             if self.save_status:
        #                 full_signal.extend(silence)
        #         # self.sleep(self.letter_gap - self.beep_gap)
        #         silence = wav.get_silence_for_duration((self.letter_gap - self.beep_gap) / 1000)
        #         if self.play_status:
        #             wav.send_signal_to_stream(silence, stream)
        #         if self.save_status:
        #             full_signal.extend(silence)

        # stream.stop_stream()
        # stream.close()
        # audio_out.terminate()

        if self.save_status:
            self.save(np.array(signal), user_input)

    def save(self, signal, text):
        assert self.save_status

        if len(signal) == 0:
            return

        fps = 44100
        max_amp = 32767
        n_frames = len(signal)

        signal *= (0.8 * max_amp) / max(abs(max(signal)), abs(min(signal)))

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with wave.open("Music/TerminalOutput/{}.wav".format(now), "w") as spf:
            spf.setnchannels(1)
            spf.setsampwidth(2)
            spf.setframerate(fps)
            spf.setnframes(n_frames)
            spf.writeframes(signal.astype("Int16").tobytes())

        with open("MusicCode/TerminalOutput/{}.txt".format(now), "w") as f:
            f.write(text)

    def sleep(self, t):
        time.sleep(t / 1000)


if __name__ == "__main__":
    terminal = MusicTerminal()
    terminal.run()