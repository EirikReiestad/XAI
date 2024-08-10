from typing import Any, Tuple
from torch.types import Number

import gymnasium as gym

from environments.gymnasium.envs.maze.utils import preprocess_state


class EnvironmentWrapper:
    """Wrapper for handling Gym environments."""

    def __init__(self, env_id: str, render_mode: str = "human"):
        self.env = gym.make(env_id, render_mode=render_mode)

    def reset(self) -> Tuple[Any, dict]:
        """Reset the environment and return the initial state and info."""
        state, info = self.env.reset()
        state = preprocess_state(state)
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
