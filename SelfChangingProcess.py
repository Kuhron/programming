# automaton-like process that includes a step to change its own numerical parameters


import random

import numpy as np

immutable_params = {
    "terminal_width": 79,
    "s": " .:!"
}

params = {
    "ema_alpha": random.random(),

}


def get(param_name):
    return params.get(param_name, immutable_params.get(param_name))


def get_random_array():
    return np.random.random(size=(1, get("terminal_width"))) - 0.5


def wrap(x):
    # symmetrical sawtooth shape so there are no discontinuous toroidal-array jumps
    # don't want to use sine wave because then distribution is too skewed to the periphery, keep it uniform in [0, 1]
    a, b = divmod(x, 1)
    return b if a % 2 == 0 else (1 - b)


def get_char(x):
    x = wrap(x)
    a = len(s)
    try:
        return s[round(a * x)]
    except:
        print(x, s, a, a*x)
        raise


if __name__ == "__main__":
    pass