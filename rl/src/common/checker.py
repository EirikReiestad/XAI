import numpy as np
import torch
from rl.src.common import getter
from gymnasium import spaces


def raise_if_not_all_same_shape_as_observation(
    a: list[np.ndarray] | list[torch.Tensor] | list[tuple[int, ...]] | list[None],
    observation_space: spaces.Space,
    name_a: str = "a",
):
    observation_mock_data = getter.get_observation_mock_data(observation_space)
    for a_ in a:
        raise_if_not_same_shape(a_, observation_mock_data, name_a, "observation")


def raise_if_not_same_shape_as_observation(
    a: np.ndarray | torch.Tensor | tuple[int, ...] | None,
    observation_space: spaces.Space,
    name_a: str = "a",
):
    observation_mock_data = getter.get_observation_mock_data(observation_space)
    raise_if_not_same_shape(a, observation_mock_data, name_a, "observation")


def raise_if_not_same_shape(
    a: np.ndarray | torch.Tensor | tuple[int, ...] | None,
    b: np.ndarray | torch.Tensor | tuple[int, ...] | None,
    name_a: str = "a",
    name_b: str = "b",
):
    if a is None:
        raise ValueError(f"{name_a} is None, but expected to be not None.")
    if b is None:
        raise ValueError(f"{name_b} is None, but expected to be not None.")

    a_np = getter.get_numpy(a)
    b_np = getter.get_numpy(b)

    a_stripped = a_np.squeeze()
    b_stripped = b_np.squeeze()

    if a_stripped.shape != b_stripped.shape:
        raise ValueError(f"Shape of {name_a}: {a_np.shape} is not {b_np.shape}.")


def raise_if_not_all_same_shape(
    a_vec: list[np.ndarray] | list[torch.Tensor],
    b: np.ndarray | torch.Tensor | tuple[int, ...] | None,
    name_a: str = "a",
    name_b: str = "b",
):
    for a in a_vec:
        raise_if_not_same_shape(a, b, name_a, name_b)
