"""Preprocess the state of the maze.
This function takes in the state of the maze and preprocesses it to be used in the DQN module."""

import torch
import numpy as np


def preprocess_state(state: np.ndarray) -> torch.Tensor:
    """
    Parameters:
        state (np.ndarray): The state of the maze
    Returns:
        torch.Tensor: The preprocessed state of the maze
    """
    torch_state = torch.tensor(state, dtype=torch.float32)
    match len(torch_state.shape):
        case 1:
            permute_state = torch_state.permute(0)
        case 2:
            permute_state = torch_state.permute(1, 0)
        case 3:
            permute_state = torch_state.permute(2, 0, 1)
        case _:
            raise ValueError(f"Invalid state shape: {torch_state.shape}")
    unsqueeze_state = permute_state.unsqueeze(0)
    return unsqueeze_state
