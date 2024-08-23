import logging
from itertools import count
from time import sleep

import matplotlib
import matplotlib.pyplot as plt
import torch

from demo import network, settings
from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from demo.src.wrappers import SingleAgentEnvironmentWrapper
from rl.src.common import ConvLayer
from rl.src.dqn.dqn_module import DQNModule


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
        env_wrapper = SingleAgentEnvironmentWrapper(env_id="MazeEnv-v0")
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
                    action = dqn.select_action(state)
                    observation, reward, terminated, truncated, _ = env_wrapper.step(
                        action.item()
                    )

                    if i_episode % settings.RENDER_EVERY == 0:
                        env_wrapper.render()

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
            return network.CONV_LAYERS
        return []


if __name__ == "__main__":
    demo = Demo()
    demo.run()