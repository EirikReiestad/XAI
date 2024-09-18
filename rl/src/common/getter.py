from typing import Union

import numpy as np
import torch
from gymnasium import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_numpy(a: np.ndarray | torch.Tensor | tuple[int, ...] | None) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    elif isinstance(a, torch.Tensor):
        if a.is_cuda:
            return a.cpu().numpy()
        return a.numpy()
    elif isinstance(a, tuple):
        return np.array(a)
    else:
        raise ValueError(f"Type of a: {type(a)} is not supported.")


def get_observation_mock_data(observation_space: spaces.Space) -> np.ndarray:
    observation_shape = get_observation_shape(observation_space)
    return np.zeros(observation_shape)


def get_observation_shape(observation_space: spaces.Space) -> tuple[int, ...]:
    observation_shape = observation_space.shape
    if observation_shape is None:
        raise ValueError("observation_space.shape is None.")
    return observation_shape


def get_same_type(
    a: Union[np.ndarray, torch.Tensor, tuple[int, ...]],
    b: Union[np.ndarray, torch.Tensor, tuple[int, ...]],
) -> Union[np.ndarray, torch.Tensor, tuple[int, ...]]:
    """Return a converted to the type of b if they are compatible."""
    if isinstance(b, np.ndarray):
        return np.array(a)
    elif isinstance(b, torch.Tensor):
        return torch.tensor(a, device=device)
    elif isinstance(b, tuple):
        if isinstance(a, np.ndarray):
            return tuple(a.tolist())
        elif isinstance(a, torch.Tensor):
            return tuple(a.numpy().tolist())
        elif isinstance(a, tuple):
            return a


def get_torch_from_numpy(a: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a
    if isinstance(a, np.ndarray):
        return torch.tensor(a, device=device, dtype=torch.float32)
    raise ValueError(f"Type of a: {type(a)} is not supported.")
