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


def raise_if_not_all_same_shape(
    a_vec: list[np.ndarray] | list[torch.Tensor],
    b: np.ndarray | torch.Tensor,
    name_a: str = "a",
    name_b: str = "b",
):
    if not all(a.shape == b.shape for a in a_vec):
        raise ValueError(
            f"All {name_a} must have shape {b.shape}, but found a mismatch."
        )
