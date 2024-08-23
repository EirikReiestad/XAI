from typing import Any, Tuple

import gymnasium as gym
import torch
from torch.types import Number

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

    @property
    def action_space(self) -> gym.spaces.Space:
        """Return the action space of the environment."""
        return self.env.action_space
