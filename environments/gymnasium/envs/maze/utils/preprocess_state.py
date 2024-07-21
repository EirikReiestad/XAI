""" Preprocess the state of the maze. 
This function takes in the state of the maze and preprocesses it to be used in the DQN module. """
import torch
import numpy as np


def preprocess_state(state: np.ndarray) -> torch.Tensor:
    """
    Parameters:
        state (np.ndarray): The state of the maze
    Returns:
        torch.Tensor: The preprocessed state of the maze
    """
    state = torch.tensor(state, dtype=torch.float32)
    state = state.permute(2, 0, 1)
    state = state.unsqueeze(0)
    return state
