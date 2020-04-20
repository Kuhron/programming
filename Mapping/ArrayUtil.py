import numpy as np


def make_nan_array(shape):
    a = np.empty(shape, dtype=float)
    a[:] = np.nan
    return a

def make_blank_condition_array(shape):
    a = np.empty(shape, dtype=object)
    a[:] = lambda x: True
    return a
