import WavUtil as w
import numpy as np
import matplotlib.pyplot as plt
import string
import random
import math


def get_click_array(length_seconds, frequency):
    threshold = 1 - frequency
    click_length = 20
    n = w.RATE * length_seconds
    arr = np.random.random(size=(n))
    binary_arr = (arr > threshold)
    for x in range(click_length):
        binary_arr |= np.roll(binary_arr, shift=x)
    # signs = (np.array([1, -1] * n))[:n]
    tone = w.get_signal_from_freq(1000, length_seconds, truncate=False)
    binary_arr = tone * binary_arr.astype("float")
    return binary_arr


def get_click_signal():
    return get_click_array(length_seconds=2, frequency=0.00025)


if __name__ == "__main__":
    n_items = 3
    signals = [get_click_signal() for _ in range(n_items)]
    names = random.sample(string.ascii_uppercase, n_items)

    # plt.plot(signal)  # sanity check
    # plt.show()
    # w.write_signal_to_wav(signal, "DolphinClick.wav")

    input("\nBeginning training period. Press enter to continue.")
    for name, signal in zip(names, signals):
        print("\nThis signal is called {}".format(name))
        for _ in range(3):
            input("\nPress enter to hear the signal{}.".format("" if _ == 0 else " again"))
            w.send_signal_to_audio_out(signal)
        input("\nPress enter to hear next signal.")

    input("\nTesting period. Identify the signal if you have heard it. Press enter if you have not.")
    score = 0
    n_trials = math.ceil(n_items * 1.5)
    indexes = list(range(n_trials))
    random.shuffle(indexes)
    for i in indexes:
        if i < n_items:
            name = names[i]
            signal = signals[i]
        else:
            signal = get_click_signal()
            name = ""
        w.send_signal_to_audio_out(signal)
        ans = input("\nWhich signal is this? ")
        display_name = name if name != "" else "a decoy signal"
        if ans == name:
            print("Correct! ")
            score += 1
        else:
            print("Whoops. You said it was {}. It was {}.".format(ans, display_name))
        input("\nPress enter to hear next signal.")

    print("\nscore = {} out of {}".format(score, n_trials))

