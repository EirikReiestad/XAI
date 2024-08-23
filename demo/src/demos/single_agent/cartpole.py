import logging
from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import torch

from demo import settings
from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from demo.src.wrappers import EnvironmentWrapper
from rl.src.dqn.dqn_module import DQNModule

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CartPoleDemo:
    """Demo for the CartPole-v1 environment."""

    def __init__(self) -> None:
        self.episode_information = EpisodeInformation(durations=[], rewards=[])
        self.plotter = Plotter()

    def run(self):
        env_wrapper = EnvironmentWrapper(env_id="CartPole-v1")
        state, _ = env_wrapper.reset()

        dqn = DQNModule(state.shape, env_wrapper.env.action_space.n)

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
            logging.error(e)
        finally:
            env_wrapper.close()
            logging.info("Complete")
            self.plotter.update(self.episode_information, show_result=True)
            plt.ioff()
        plt.show()


if __name__ == "__main__":
    demo = CartPoleDemo()
    demo.run()