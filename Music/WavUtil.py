import pyaudio
import wave
import numpy as np


RATE = 44100
MAX_AMPLITUDE = 32767


def get_signal_from_freq(freq, seconds, initial_click=False):
    n_frames = RATE * seconds
    n_frames_truncated_at_phase_zero = n_frames - (n_frames % (RATE / freq))
    xs = np.arange(n_frames_truncated_at_phase_zero)
    ys = np.sin(freq * 2*np.pi / RATE * xs) * 10

    if initial_click:
        ys = np.array([10, -10] * 4 + list(ys))

    return ys


def get_signal_from_notes(notes):
    res = []
    for note in notes:
        res.extend(list(note.get_wav_signal()))
    return np.array(res)


def get_silence_for_duration(seconds):
    n_frames = RATE * seconds
    xs = np.arange(n_frames)
    ys = xs * 0
    return ys


def send_freq_to_stream(freq, seconds, stream, initial_click=False):
    ys = get_signal_from_freq(freq, seconds, initial_click=False)
    send_signal_to_stream(ys, stream)


def send_signal_to_stream(ys, stream):
    stream.write(ys.astype("Int8").tobytes())


def send_signal_to_audio_out(signal):
    audio_out = pyaudio.PyAudio()
    stream = audio_out.open(
        format=pyaudio.paInt8,
        channels=1,
        rate=RATE,
        output=True,
    )

    send_signal_to_stream(signal, stream)

    stream.stop_stream()
    stream.close()
    audio_out.terminate()


def write_signal_to_wav(signal, filepath):
    with wave.open(filepath, "w") as spf:
        spf.setnchannels(1)
        spf.setsampwidth(2)
        spf.setframerate(RATE)
        spf.setnframes(len(signal))
        signal *= (0.8 * MAX_AMPLITUDE) / max(abs(max(signal)), abs(min(signal)))
        spf.writeframes(signal.astype("Int16").tobytes())
