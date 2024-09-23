import numpy as np


def get_q_values_map(
    states: np.ndarray, q_values: np.ndarray, max_q_values: bool = False
) -> np.ndarray:
    if states.shape[:2] != q_values.shape[:2]:
        raise ValueError(
            f"States shape {states.shape[:2]} does not match Q-values shape {q_values.shape[:2]}"
        )

    if max_q_values:
        max_q_values = np.max(q_values, axis=2)
        normalized_max_q_values = (max_q_values - np.min(max_q_values)) / np.ptp(
            max_q_values
        )
        return normalized_max_q_values

    adjusted_q_values = q_values + np.abs(np.min(q_values))
    normalized_q_values = (adjusted_q_values - np.min(adjusted_q_values)) / np.ptp(
        adjusted_q_values
    )
    cumulated_q_values = np.zeros((states.shape[0], states.shape[1]), dtype=np.float32)
    cumulated_q_values_count = np.zeros(
        (states.shape[0], states.shape[1]), dtype=np.int32
    )

    width, height = states.shape[:2]
    for y in range(height):
        for x in range(width):
            # We use the following order: up, down, left, right
            if x > 0:
                cumulated_q_values[y, x - 1] += normalized_q_values[y, x, 2]
                cumulated_q_values_count[y, x - 1] += 1
            if x < height - 1:
                cumulated_q_values[y, x + 1] += normalized_q_values[y, x, 3]
                cumulated_q_values_count[y, x + 1] += 1
            if y > 0:
                cumulated_q_values[y - 1, x] += normalized_q_values[y, x, 0]
                cumulated_q_values_count[y - 1, x] += 1
            if y < width - 1:
                cumulated_q_values[y + 1, x] += normalized_q_values[y, x, 1]
                cumulated_q_values_count[y + 1, x] += 1

    for y in range(height):
        for x in range(width):
            if cumulated_q_values_count[y, x] > 0:
                cumulated_q_values[y, x] /= cumulated_q_values_count[y, x]

    normalized_cumulated_q_values = (
        cumulated_q_values - np.min(cumulated_q_values)
    ) / np.ptp(cumulated_q_values)

    return normalized_cumulated_q_values
