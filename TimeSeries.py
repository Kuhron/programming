import random

import numpy as np
import scipy
import matplotlib.pyplot as plt


def get_damped_linear_lookback_process(n_terms, n_lookback):
    seed = np.random.rand(n_lookback)
    weights = np.random.uniform(-2, 2, n_lookback)

    damping_power = random.uniform(0, 1)

    result = np.array(list(seed) + [None for i in range(n_terms)])
    for i in range(n_terms):
        new_term = sum(result[i: n_lookback + i] * weights)
        new_term = np.sign(new_term) * (abs(new_term) ** damping_power)
        result[n_lookback + i] = new_term

    return result[n_lookback:]


def get_set_augmentation_process(n_terms):
    result = []
    for i in range(n_terms):
        if i == 0:
            new = random.uniform(-1, 1)
        else:
            new = random.choice(result) + random.uniform(-1, 1)
        result.append(new)
    return result


def get_ema_from_process(series, alpha=None):
    if alpha is None:
        alpha = random.random()
    result = []
    for i in range(len(series)):
        if i == 0:
            new = series[i]
        else:
            new = alpha * series[i] + (1 - alpha) * result[i - 1]
        result.append(new)
    return result


def get_nth_ema_from_process(series, order, alpha=None):
    if order <= 1:
        return get_ema_from_process(series, alpha)
    return get_ema_from_process(get_nth_ema_from_process(series, order - 1, alpha), alpha)


def get_cumsum_from_process(series):
    return np.array(series).cumsum()



series = None
for i in range(5):
    ser = get_set_augmentation_process(10000)
    ser = [int(x * 1000) for x in ser]
    # ser = get_nth_ema_from_process(ser, 2, 0.1)
    if i == 0:
        series = np.array(ser)
    else:
        series ^= np.array(ser)

series = get_cumsum_from_process(series)
series = get_ema_from_process(series, 0.5)

plt.plot(series)
plt.show()