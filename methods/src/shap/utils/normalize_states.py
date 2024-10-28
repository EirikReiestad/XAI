import numpy as np


def normalize_states(states: np.ndarray, a: float = 0, b: float = 1) -> np.ndarray:
    global_min = np.min(states)
    global_max = np.max(states)

    states = (states - global_min) / (global_max - global_min) * (b - a) + a

    return states
