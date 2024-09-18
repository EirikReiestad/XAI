import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_state(state: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Preprocesses the state of the maze to be compatible with the DQN module.

    Args:
        state (np.ndarray): The state of the maze.

    Returns:
        torch.Tensor: The preprocessed state of the maze.
    """
    # Convert the NumPy array to a PyTorch tensor
    if isinstance(state, np.ndarray):
        torch_state = torch.tensor(state, device=device, dtype=torch.float32)
    else:
        torch_state = state

    # Permute dimensions based on the tensor's shape
    if torch_state.ndim == 1:
        permute_state = torch_state
    elif torch_state.ndim == 2:
        permute_state = torch_state.permute(0, 1)
    elif torch_state.ndim == 3:
        permute_state = torch_state.permute(2, 0, 1)
    else:
        raise ValueError(f"Invalid state shape: {torch_state.shape}")

    # Add a batch dimension
    return permute_state.unsqueeze(0)
