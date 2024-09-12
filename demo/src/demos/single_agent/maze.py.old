from itertools import count
from typing import Optional

import torch

from demo import settings
from demo.src.common import Batch, Transition
from demo.src.demos.single_agent import BaseDemo
from demo.src.wrappers.single_agent_environment_wrapper import (
    SingleAgentEnvironmentWrapper,
)


class MazeDemo(BaseDemo):
    """Class for running the Maze demo with DQN and plotting results."""

    def _create_environment_wrapper(
        self, render_mode: Optional[str] = None
    ) -> SingleAgentEnvironmentWrapper:
        """Create and return the environment wrapper specific to the Maze demo."""
        return SingleAgentEnvironmentWrapper(
            env_id="MazeEnv-v0", render_mode=render_mode
        )

    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict) -> Batch:
        state, _ = self.env_wrapper.reset()
        total_reward = 0

        batch = Batch(
            states=[],
            actions=[],
            observations=[],
            rewards=[],
            terminated=[],
            truncated=[],
        )

        for t in count():
            action = self.dqn.select_action(state)
            observation, reward, terminated, truncated, _ = self.env_wrapper.step(
                action.item()
            )

            reward = float(reward)
            total_reward += reward

            done = terminated or truncated

            transition = Transition(
                state=state,
                action=action,
                observation=observation,
                reward=torch.tensor([reward], dtype=torch.float32),
                terminated=terminated,
                truncated=truncated,
            )
            batch.append(transition)

            if i_episode % settings.RENDER_EVERY == 0:
                self.render()

            if done:
                self.episode_information.durations.append(t + 1)
                self.episode_information.rewards.append(total_reward)
                if self.plotter is not None:
                    self.plotter.update(self.episode_information)
                break

        return batch
