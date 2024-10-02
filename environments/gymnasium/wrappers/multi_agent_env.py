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
        full_state, concatenated_state_rewards, terminated, concat_truncated = (
            self.get_wrapper_attr("concatenate_states")(full_states)
        )
        observation = self.get_wrapper_attr("update_state")(full_state)

        total_rewards = [r + cr for r, cr in zip(rewards, concatenated_state_rewards)]

        done = any(terminals) or terminated or concat_truncated or any(truncated)
        if total_rewards != [0.0, 0.0]:
            print("rewards", total_rewards, rewards, concatenated_state_rewards, done)
        return (
            full_state,
            observation,
            done,
            observations,
            total_rewards,
            terminals,
            truncated,
            infos,
        )
