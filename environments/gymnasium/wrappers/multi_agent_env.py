from typing import Any

import gymnasium as gym
import numpy as np
from torch.types import Number


class MultiAgentEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        _, info = self.env.reset()
        self.concatenated_states_fn = info.get("concatenated_states_fn")

    def step_multiple(
        self, actions: list[Number]
    ) -> tuple[
        np.ndarray,
        bool,
        list[np.ndarray],
        list[float],
        list[bool],
        list[bool],
        list[dict[str, Any]],
    ]:
        observations, rewards, terminals, truncated, infos = [], [], [], [], []
        for action in actions:
            observation, reward, terminated, trunc, info = self.env.step(action)
            observations.append(observation)
            rewards.append(reward)
            terminals.append(terminated)
            truncated.append(trunc)
            infos.append(info)
        concatenated_state, concatenated_state_rewards, terminated = (
            self.concatenate_states_fn(observations)
        )
        rewards += concatenated_state_rewards
        return (
            concatenated_state,
            terminated,
            observations,
            rewards,
            terminals,
            truncated,
            infos,
        )

    def _reset(self):
        _, info = self.env.reset()
        self.concatenated_states_fn = info.get("concatenated_states_fn")
        if not callable(self.concatenated_states_fn):
            raise ValueError("The concatenated_states_fn is not a callable function.")
