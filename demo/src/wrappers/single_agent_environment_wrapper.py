from typing import Any, Optional, Tuple

import gymnasium as gym
import torch
from torch.types import Number
import numpy as np

from environments.gymnasium.utils import preprocess_state


class SingleAgentEnvironmentWrapper:
    """Wrapper for handling Gym environments."""

    def __init__(self, env_id: str, render_mode: str = "human"):
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

    def render(self):
        """Render the environment."""
        self.env.render()

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

        if all_possible_states is None or any([state is None for state in all_possible_states]) is None:
            raise ValueError("All possible states must not contain any None values.")
        if not isinstance(all_possible_states, np.ndarray):
            raise ValueError(f"All possible states must be a NumPy array, not {type(all_possible_states)}")

        if all_possible_states.shape != self.env.observation_space.shape:
            raise ValueError(
                f"All possible states must have the same shape as the environment's observation space. Got {all_possible_states.shape}, expected {self.env.observation_space.shape}."
            )

        states = np.ndarray(all_possible_states.shape, dtype=torch.Tensor)
        for y, row_state in enumerate(all_possible_states):
            for x, state in enumerate(row_state):
                states[x, y] = preprocess_state(state)

        return states

    def render_q_values(self, q_values: np.ndarray):
        """Get the Q-values for a given state."""
        self.env.unwrapped.render_q_values(q_values)

    @property
    def action_space(self) -> gym.spaces.Space:
        """Return the action space of the environment."""
        return self.env.action_space
