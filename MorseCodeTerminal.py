import datetime
import random
import pyaudio
import string
import time
import wave
import readline  # press up to be able to get previously typed terminal input

import numpy as np

import Music.WavUtil as wav
import Terminal


MORSE_TABLE = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    "!": "-.-.--",
    "?": "..--..",
    ",": "--..--",
    "\'": ".----.",
    " ": " ",
}


class MorseCodeTerminal(Terminal.Terminal):
    def __init__(self):
        super().__init__()

        self.dit = 1
        self.dah = 3

        self.update_wpm(35)

        self.tone_hz = 2**(np.log2(440) +3/12 -12/12)

        self.initialize_commands()
        self.save_status = False
        self.initial_click = False
        self.play_status = True
        self.word_by_word = True

    def initialize_commands(self):
        super().initialize_commands()
        
        self.add_command("wpm", self.process_wpm_input, "Change wpm to arg if given, else show current wpm.")
        self.add_command("s", self.change_save_status, "Turn on saving to file if arg is 1, else turn off if arg is 0.")
        self.add_command("c", self.change_initial_click, "Turn on initial click on each beep if arg is 1, else turn off if arg is 0.")
        self.add_command("p", self.change_play_status, "Turn on out-loud play if arg is 1, else turn off if arg is 0.")
        self.add_command("rand", self.play_random_sentence, "Play a random sequence of {arg} words.")
        self.add_command("hz", self.process_hz_input, "Change tone frequency to arg if given, else show current tone frequency in Hz.")
        self.add_command("repeat", self.repeat, "Repeat the input (args) until interrupted by Ctrl+C")
        self.add_command("wbw", self.change_word_by_word, "Turn on word-by-word printing if arg is 1, else turn off if arg is 0.")

    def change_save_status(self, a=None):
        return self.change_binary_attribute("save_status", a)

    def change_initial_click(self, a=None):
        return self.change_binary_attribute("initial_click", a)

    def change_play_status(self, a=None):
        return self.change_binary_attribute("play_status", a)

    def change_word_by_word(self, a=None):
        return self.change_binary_attribute("word_by_word", a)

    def process_normal_input(self, user_input):
        self.play(user_input)

    def process_wpm_input(self, wpm=None):
        if wpm is not None:
            self.update_wpm(int(wpm))
            return
        else:
            return self.wpm

    def update_wpm(self, wpm):
        # https://en.wikipedia.org/wiki/Morse_code#Speed_in_words_per_minute

        self.wpm = int(wpm)

        self.dit = int(1200 / self.wpm * 1)
        self.dah = int(1200 / self.wpm * 3)

        self.beep_gap = self.dit
        self.letter_gap = self.dah
        self.word_gap = int(1200 / self.wpm * 7)

    def process_hz_input(self, hz=None):
        if hz is not None:
            self.update_hz(int(hz))
            return
        else:
            return self.tone_hz

    def update_hz(self, hz):
        self.tone_hz = hz

    def play_random_sentence(self, n_words=None):
        corpus_fp = random.choice(["epictetus-discourses.txt", "war-and-peace.txt", "frankenstein.txt", "moby-dick.txt", "jekyll-and-hyde.txt", "dracula.txt", "monte-cristo.txt"])
        with open(corpus_fp) as f:
            contents = f.read()
        split = contents.split(" ")
        words = []
        start_index = random.randrange(len(split)-100)
        n_words_to_get = 10 if n_words is None else int(n_words)
        i = start_index
        while True:
            w = split[i].strip()
            if w != "":
                words.append(w)
            if len(words) >= n_words_to_get:
                break
            if i >= len(split):
                break
            i += 1
        sentence = " ".join(words).replace("\n", " ")
        print("\n"+sentence+"\n")
        self.play(sentence)

    def translate(self, s):
        s = [x for x in s.upper() if x in string.ascii_uppercase + "0123456789., "]
        morse = [MORSE_TABLE[x] for x in s]
        return morse

    def pad_morse(self, morse):
        return [" "]*2 + morse + [" "]*2

    def repeat(self, *user_input):
        user_input = " ".join(user_input)
        while True:
            self.play(user_input)

    def play(self, user_input, pad=True):
        audio_out = pyaudio.PyAudio()
        rate = 44100
        stream = audio_out.open(
            format=pyaudio.paInt8,
            channels=1,
            rate=rate,
            output=True,
            frames_per_buffer=256,
        )

        full_signal = []
        morse = self.translate(user_input)
        if pad:
            morse = self.pad_morse(morse)

        if self.word_by_word:
            morse_words = list_split(morse, " ", retain_delimiter=True)
            user_input_words = user_input.split(" ")
            # make padding match
            for wi, mw in enumerate(morse_words):
                if mw == [" "]:
                    user_input_words = [" "] + user_input_words
                else:
                    break
            for wi, mw in enumerate(morse_words[::-1]):
                if mw == [" "]:
                    user_input_words = user_input_words + [" "]
                else:
                    break
            assert len(morse_words) == len(user_input_words), "{}\n!=\n{}".format(morse_words, user_input_words)
        else:
            morse_words = morse

        for wi, morse_word in enumerate(morse_words):
            if self.word_by_word and morse_word != [" "]:
                print("-- " + user_input_words[wi])
            for morse_letter in morse_word:
                if morse_letter == " ":
                    silence = wav.get_silence_for_duration((self.word_gap - self.letter_gap) / 1000)
                    wav.send_signal_to_stream(silence, stream)
                    if self.save_status:
                        full_signal.extend(silence)
                else:
                    durations = [self.dit if morse_tone == "." else self.dah for morse_tone in morse_letter]
                    for duration in durations:
                        # winsound.Beep(self.tone_hz, duration)
                        signal = wav.get_signal_from_freq(self.tone_hz, duration / 1000, initial_click=self.initial_click)
                        if self.play_status:
                            wav.send_signal_to_stream(signal, stream)
                        if self.save_status:
                            full_signal.extend(signal)
                        # self.sleep(self.beep_gap)
                        silence = wav.get_silence_for_duration(self.beep_gap / 1000)
                        if self.play_status:
                            wav.send_signal_to_stream(silence, stream)
                        if self.save_status:
                            full_signal.extend(silence)
                    # self.sleep(self.letter_gap - self.beep_gap)
                    silence = wav.get_silence_for_duration((self.letter_gap - self.beep_gap) / 1000)
                    if self.play_status:
                        wav.send_signal_to_stream(silence, stream)
                    if self.save_status:
                        full_signal.extend(silence)

        stream.stop_stream()
        stream.close()
        audio_out.terminate()

        if self.save_status:
            self.save(np.array(full_signal), user_input)

    def save(self, signal, text):
        assert self.save_status

        if len(signal) == 0:
            return

        fps = 44100
        max_amp = 32767
        n_frames = len(signal)

        signal *= (0.8 * max_amp) / max(abs(max(signal)), abs(min(signal)))

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with wave.open("MorseCode/TerminalOutput/{}.wav".format(now), "w") as spf:
            spf.setnchannels(1)
            spf.setsampwidth(2)
            spf.setframerate(fps)
            spf.setnframes(n_frames)
            spf.writeframes(signal.astype("Int16").tobytes())

        with open("MorseCode/TerminalOutput/{}.txt".format(now), "w") as f:
            f.write(text)

    def sleep(self, t):
        time.sleep(t / 1000)


def list_split(lst, delim, retain_delimiter=False):
    res = []
    current_sublst = []
    for item in lst:
        if item == delim:
            if retain_delimiter:
                current_sublst.append(item)
            res.append(current_sublst)
            current_sublst = []
        else:
            current_sublst.append(item)
    if current_sublst != []:
        res.append(current_sublst)
    return res



if __name__ == "__main__":
    terminal = MorseCodeTerminal()
    terminal.run()
