import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def to_arr(value):
    if isinstance(value, list) or isinstance(value, np.ndarray):
        return value
    return [value]
