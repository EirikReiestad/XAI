import numpy as np


def normalize_states(states: np.ndarray) -> np.ndarray:
    a = 0
    b = 255

    global_min = np.min(states)
    global_max = np.max(states)

    states = (states - global_min) / (global_max - global_min) * (b - a) + a

    return states
