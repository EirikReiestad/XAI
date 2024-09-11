import numpy as np
import torch


def raise_if_not_same_shape(
    a: np.ndarray | torch.Tensor,
    b: np.ndarray | torch.Tensor | tuple[int, ...] | None,
    name_a: str = "a",
    name_b: str = "b",
):
    if b is None:
        raise ValueError(f"{name_b} is None, but expected to be not None.")
    if isinstance(b, tuple):
        if a.shape != b:
            raise ValueError(f"Shape of {name_a}: {a.shape} is not {b}.")
    else:
        if a.shape != b.shape:
            raise ValueError(
                f"Shape of {name_a}: {a.shape} is not same as {name_b}: {b.shape}"
            )


def raise_if_not_all_same_shape(
    a_vec: list[np.ndarray] | list[torch.Tensor],
    b: np.ndarray | torch.Tensor | tuple[int, ...] | None,
    name_a: str = "a",
    name_b: str = "b",
):
    for a in a_vec:
        raise_if_not_same_shape(a, b, name_a, name_b)
