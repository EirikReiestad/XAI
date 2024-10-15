import getpass
import logging
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch

from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from methods import Shap
from rl.src.dqn import DQN
from rl.src.managers import WandBConfig

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CartPoleDemo:
    """Demo for the CartPole-v1 environment."""

    def __init__(self) -> None:
        self.episode_information = EpisodeInformation(
            durations=[], rewards=[], object_moved_distance=[]
        )
        self.plotter = Plotter(plot_agent=True)
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        current_user = getpass.getuser()
        project = f"tag-v0-{current_user}"
        wandb_config = WandBConfig(project=project)
        self.dqn = DQN(
            self.env,
            "dqnpolicy",
            wandb_active=True,
            save_model=False,
            wandb_config=wandb_config,
            gif=True,
        )

    def run(self):
        logging.info("Learning...")
        self.dqn.learn(500)

        self.shape(False)
        self.show(False)

    def shape(self, show: bool = False):
        if not show:
            return

        logging.info("Initializing Shap...")
        self.shap = Shap(self.env, self.dqn)
        logging.info("Explaining...")
        shap_values = self.shap.explain()
        self.shap.plot(
            shap_values,
            feature_names=[
                "Cart Position",
                "Cart Velocity",
                "Pole Angle",
                "Pole Velocity",
            ],
        )

    def show(self, run: bool = True):
        if not run:
            return

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
                    action = self.dqn.predict_action(state)
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
