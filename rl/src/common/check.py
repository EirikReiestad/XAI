import numpy as np
import torch


def raise_if_not_same_shape(
    a: np.ndarray | torch.Tensor,
    b: np.ndarray | torch.Tensor,
    name_a: str = "a",
    name_b: str = "b",
):
    if a.shape != b.shape:
        raise ValueError(
            f"Shape of {name_a}: {a.shape} is not same as {name_b}: {b.shape}"
        )
