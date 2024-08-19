from typing import Any, Tuple

import gymnasium as gym
import torch
from torch.types import Number

from environments.gymnasium.envs.maze.utils import preprocess_state


class MultiAgentEnvironmentWrapper:
    """Wrapper for handling Multi-Agent environments."""

    def __init__(self, env_id: str, render_mode: str = "human"):
        self.env = gym.make(env_id, render_mode=render_mode)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """Reset the environment and return the initial state and info."""
        state, info = self.env.reset()
        state = preprocess_state(state)
        if not isinstance(state, torch.Tensor):
            raise ValueError("State must be a PyTorch tensor after preprocessing.")
        return state, info

    def step(
        self, actions: list[Number]
    ) -> Tuple[list[Any], list[float], list[bool], list[bool]]:
        """Perform an action in the environment and return the results."""
        observations = []
        rewards = []
        terminated = []
        truncated = []
        for action in actions:
            observation, reward, term, trunc, _ = self.env.step(action)
            observation = preprocess_state(observation)
            observations.append(observation)
            rewards.append(float(reward))
            terminated.append(term)
            truncated.append(trunc)

        return observations, rewards, terminated, truncated

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
