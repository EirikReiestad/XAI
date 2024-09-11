import numpy as np


def raise_if_not_same_shape(
    a: np.ndarray, b: np.ndarray, name_a: str = "a", name_b: str = "b"
):
    if a.shape != b.shape:
        raise ValueError(
            f"Shape of {name_a}: {a.shape} is not same as {name_b}: {b.shape}"
        )
