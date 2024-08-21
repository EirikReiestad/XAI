from typing import Any, Tuple

import gymnasium as gym
import torch
from torch.types import Number

from environments.gymnasium.utils import preprocess_state


class EnvironmentWrapper:
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

    def step(self, action: Number) -> Tuple[Any, float, bool, bool]:
        """Perform an action in the environment and return the results."""
        observation, reward, terminated, truncated, _ = self.env.step(action)
        observation = preprocess_state(observation)
        return observation, float(reward), terminated, truncated

    def render(self):
        """Render the environment."""
        self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    def concat_state(self, states: list[torch.Tensor]) -> torch.Tensor:
        """Return the concatenated state of the environment."""
        numpy_states = [state.numpy() for state in states]
        state = self.env.unwrapped.concat_state(numpy_states)
        tensor_state = torch.tensor(state)
        return tensor_state

    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return self.env.num_agents

    @property
    def action_space(self) -> gym.spaces.Space:
        """Return the action space of the environment."""
        return self.env.action_space
