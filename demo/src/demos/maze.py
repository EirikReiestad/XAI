import logging
from itertools import count
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo import settings
from demo.src.common import EpisodeInformation, episode_information
from environments.gymnasium.envs.maze.utils import preprocess_state
from rl.src.common import ConvLayer
from rl.src.dqn.dqn_module import DQNModule
from demo.src.plotters import Plotter
from demo.src.wrappers import EnvironmentWrapper

# Register Gym environment
gym.register(
    id="Maze-v0",
    entry_point="environments.gymnasium.envs.maze.maze:MazeEnv",
)


class Demo:
    """Class for running the Maze demo with DQN and plotting results."""

    def __init__(self):
        """Initialize the Demo class with settings and plotter."""
        self.episode_information = EpisodeInformation(durations=[], rewards=[])
        self.plotter = Plotter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_ipython = "inline" in matplotlib.get_backend()

    def run(self):
        """Run the demo, interacting with the environment and training the DQN."""
        env_wrapper = EnvironmentWrapper(env_id="Maze-v0")
        state, info = env_wrapper.reset()
        n_actions = env_wrapper.action_space.n
        conv_layers = self._create_conv_layers(info)

        dqn = DQNModule(state.shape, n_actions, conv_layers=conv_layers)
        plt.ion()

        try:
            for i_episode in range(settings.NUM_EPISODES):
                state, _ = env_wrapper.reset()
                total_reward = 0

                for t in count():
                    if i_episode % settings.RENDER_EVERY == 0:
                        env_wrapper.render()

                    action = dqn.select_action(state)
                    observation, reward, terminated, truncated = env_wrapper.step(
                        action.item()
                    )
                    reward = float(reward)

                    total_reward += reward

                    done, new_state = dqn.train(
                        state, action, observation, reward, terminated, truncated
                    )

                    state = new_state if not done and new_state is not None else state

                    if done:
                        self.episode_information.durations.append(t + 1)
                        self.episode_information.rewards.append(total_reward)
                        self.plotter.update(self.episode_information)
                        break

        except Exception as e:
            logging.exception(e)
        finally:
            env_wrapper.close()
            logging.info("Complete")
            self.plotter.update(self.episode_information, show_result=True)
            plt.ioff()
            plt.show()

    def _create_conv_layers(self, info) -> list[ConvLayer]:
        """Create convolutional layers based on the state type."""
        state_type = info.get("state_type") if info else None
        if state_type in {"rgb", "full"}:
            return [
                ConvLayer(
                    filters=32,
                    kernel_size=2,
                    strides=2,
                    activation="relu",
                    padding="same",
                ),
                ConvLayer(
                    filters=32,
                    kernel_size=2,
                    strides=2,
                    activation="relu",
                    padding="same",
                ),
                ConvLayer(
                    filters=32,
                    kernel_size=2,
                    strides=2,
                    activation="relu",
                    padding="same",
                ),
            ]
        return []


if __name__ == "__main__":
    demo = Demo()
    demo.run()
