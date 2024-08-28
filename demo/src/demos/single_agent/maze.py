import torch
from itertools import count
from demo import settings
from demo.src.demos.single_agent import BaseDemo
from demo.src.wrappers.single_agent_environment_wrapper import (
    SingleAgentEnvironmentWrapper,
)


class MazeDemo(BaseDemo):
    """Class for running the Maze demo with DQN and plotting results."""

    def _create_environment_wrapper(self) -> SingleAgentEnvironmentWrapper:
        """Create and return the environment wrapper specific to the Maze demo."""
        return SingleAgentEnvironmentWrapper(env_id="MazeEnv-v0")

    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict):
        state, _ = self.env_wrapper.reset()
        total_reward = 0
        for t in count():
            action = self.dqn.select_action(state)
            observation, reward, terminated, truncated, _ = self.env_wrapper.step(
                action.item()
            )

            if i_episode % settings.RENDER_EVERY == 0:
                # self.env_wrapper.render()
                if settings.QVALUES:
                    self._render_q_values()
                else:
                    self.env_wrapper.render()

            reward = float(reward)
            total_reward += reward

            done, new_state = self.dqn.train(
                state, action, observation, reward, terminated, truncated
            )

            state = new_state if not done and new_state is not None else state

            if done:
                self.episode_information.durations.append(t + 1)
                self.episode_information.rewards.append(total_reward)
                if self.plotter is not None:
                    self.plotter.update(self.episode_information)
                break
