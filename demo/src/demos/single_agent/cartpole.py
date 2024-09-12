import logging
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch

from demo import settings
from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from rl.src.dqn import DQN

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CartPoleDemo:
    """Demo for the CartPole-v1 environment."""

    def __init__(self) -> None:
        self.episode_information = EpisodeInformation(
            durations=[], rewards=[], object_moved_distance=[]
        )
        self.plotter = Plotter()
        self.env = gym.make("CartPole-v1")

    def run(self):
        dqn = DQN("dqnpolicy", self.env, wandb=True)
        dqn.learn(settings.EPOCHS)
        print("Training complete")

        self.env = gym.make("CartPole-v1", render_mode="human")

        plt.ion()

        self.env.reset()

        try:
            for i_episode in range(1000):
                state, _ = self.env.reset()
                state = torch.tensor(
                    state, device=device, dtype=torch.float32
                ).unsqueeze(0)

                rewards = 0

                for t in count():
                    action = dqn.predict(state)
                    observation, reward, terminated, truncated, _ = self.env.step(
                        action.item()
                    )
                    state = torch.tensor(
                        observation, device=device, dtype=torch.float32
                    ).unsqueeze(0)
                    self.env.render()
                    rewards += float(reward)

                    if terminated or truncated:
                        self.episode_information.durations.append(t + 1)
                        self.episode_information.rewards.append(rewards)
                        self.plotter.update(self.episode_information)
                        break
        except Exception as e:
            logging.error(e)
        finally:
            self.env.close()
            logging.info("Complete")
            self.plotter.update(self.episode_information, show_result=True)
            plt.ioff()
        plt.show()


if __name__ == "__main__":
    demo = CartPoleDemo()
    demo.run()
