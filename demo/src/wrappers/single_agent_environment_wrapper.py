from typing import Any, Optional, Tuple

import gymnasium as gym
import torch
from torch.types import Number
import numpy as np

from environments.gymnasium.utils import preprocess_state


class SingleAgentEnvironmentWrapper:
    """Wrapper for handling Gym environments."""

    def __init__(self, env_id: str, render_mode: Optional[str] = "human"):
        self.env = gym.make(env_id, render_mode=render_mode)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """Reset the environment and return the initial state and info."""
        state, info = self.env.reset()
        state = preprocess_state(state)
        if not isinstance(state, torch.Tensor):
            raise ValueError("State must be a PyTorch tensor after preprocessing.")
        return state, info

    def step(self, action: Number) -> Tuple[Any, float, bool, bool, dict]:
        """Perform an action in the environment and return the results."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = preprocess_state(observation)
        return observation, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        result = self.env.render()
        if isinstance(result, list):
            if len(result) == 0:
                return result[0]
            else:
                raise ValueError("Rendering must return a single NumPy array.")
        elif isinstance(result, np.ndarray) or result is None:
            return result
        else:
            raise TypeError("Unexpected return type from env.render().")

    def close(self):
        """Close the environment."""
        self.env.close()

    def get_all_possible_states(self) -> np.ndarray:
        """Get all possible states for the agent in the environment."""
        options = {
            "all_possible_states": True,
        }
        _, info = self.env.reset(options=options)
        all_possible_states = info.get("all_possible_states")

        if (
            all_possible_states is None
            or any([state is None for state in all_possible_states]) is None
        ):
            raise ValueError("All possible states must not contain any None values.")
        if not isinstance(all_possible_states, np.ndarray):
            raise ValueError(
                f"All possible states must be a NumPy array, not {type(all_possible_states)}"
            )

        states = np.ndarray(all_possible_states.shape[:2], dtype=torch.Tensor)
        for y, row_state in enumerate(all_possible_states):
            for x, state in enumerate(row_state):
                torch_state = preprocess_state(state)
                states[y, x] = torch_state
        return states

    @property
    def action_space(self) -> gym.spaces.Space:
        """Return the action space of the environment."""
        return self.env.action_space