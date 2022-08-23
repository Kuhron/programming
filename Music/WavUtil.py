import pyaudio
import wave
import numpy as np
from scipy.io.wavfile import read as read_wav


RATE = 44100
MAX_AMPLITUDE = 32767


def get_signal_from_freq(freq, seconds, initial_click=False, truncate="backward", spectrum=None):
    n_frames = RATE * seconds

    frames_per_sec = RATE
    cycles_per_sec = freq
    frames_per_cycle = RATE / freq
    if truncate == "backward":
        n_frames_truncated_at_phase_zero = n_frames - (n_frames % frames_per_cycle)
        xs = np.arange(n_frames_truncated_at_phase_zero)
    elif truncate == "forward":
        # go forward to the next phase-zero point instead of truncating backward
        remainder_hanging_off_right_edge = n_frames % frames_per_cycle
        if remainder_hanging_off_right_edge > 0:
            remaining_frames_in_this_cycle = frames_per_cycle - remainder_hanging_off_right_edge
            n_frames_truncated_at_phase_zero = n_frames + remaining_frames_in_this_cycle
            xs = np.arange(n_frames_truncated_at_phase_zero)
        else:
            # don't add another cycle if we're already ending at phase zero
            xs = np.arange(n_frames)
    else:
        xs = np.arange(n_frames)
        
    ys = np.sin(freq * 2*np.pi / RATE * xs) * 10

    if initial_click:
        ys = np.array([10, -10] * 4 + list(ys))

    return ys


def get_signal_from_notes(notes, spectrum=None):
    res = []
    for note in notes:
        res.extend(list(note.get_wav_signal(spectrum=spectrum)))
    return np.array(res)


def get_silence_for_duration(seconds):
    n_frames = RATE * seconds
    xs = np.arange(n_frames)
    ys = xs * 0
    return ys


def pad_signal_with_silence(signal, seconds_per_side):
    sil = get_silence_for_duration(seconds_per_side)
    signal = np.concatenate([sil, signal, sil])
    return signal


def send_freq_to_stream(freq, seconds, stream, initial_click=False):
    ys = get_signal_from_freq(freq, seconds, initial_click=False)
    send_signal_to_stream(ys, stream)


def send_signal_to_stream(ys, stream):
    ys_bytes = ys.astype("Int8").tobytes()

    # split into segments so it's not all sent at once, blocking the program from exiting with Ctrl-C
    seg_length = 1024  # even a few-second wav signal is ~50k
    while True:
        stream.write(ys_bytes[:seg_length])
        ys_bytes = ys_bytes[seg_length:]
        if len(ys_bytes) == 0:
            return


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


def write_signal_to_wav(signal, filepath, rate=RATE, amplitude_fraction=0.8):
    assert 0 < amplitude_fraction <= 1, amplitude_fraction
    signal = np.array(signal)
    desired_amplitude = amplitude_fraction * MAX_AMPLITUDE
    max_abs = abs(signal).max()
    if max_abs > 0:
        signal *= desired_amplitude / max_abs
    with wave.open(filepath, "w") as spf:
        spf.setnchannels(1)
        spf.setsampwidth(2)
        spf.setframerate(rate)
        spf.setnframes(len(signal))
        spf.writeframes(signal.astype("Int16").tobytes())


def read_wav_to_array(filepath):
    f = read_wav(filepath)
    return np.array(f[1], dtype=float)
