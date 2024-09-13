from typing import Any

import gymnasium as gym
import numpy as np
from torch.types import Number


class MultiAgentEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        _, info = self.env.reset()

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

        full_states = [info["full_state"] for info in infos]
        concatenated_state, concatenated_state_rewards, terminated = (
            self.get_wrapper_attr("concatenate_states")(full_states)
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
