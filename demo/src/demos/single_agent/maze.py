import logging
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch

from rl.src.managers import WandBConfig

from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from rl.src.dqn import DQN

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MazeDemo:
    """Demo for the CartPole-v1 environment."""

    def __init__(self) -> None:
        self.episode_information = EpisodeInformation(
            durations=[], rewards=[], object_moved_distance=[]
        )
        self.plotter = Plotter(plot_agent=True)
        self.env = gym.make("MazeEnv-v0", render_mode="rgb_array")
        wandb_config = WandBConfig(project="maze-v0-local")
        self.dqn = DQN(
            self.env,
            "dqnpolicy",
            wandb_active=True,
            wandb_config=wandb_config,
            save_model=True,
            load_model=False,
            run_path="eirikreiestad-ntnu/maze-v0-local",
            model_artifact="model_200",
        )

    def run(self, num_episodes=1000):
        self.dqn.learn(num_episodes)

        self.show(False)

    def show(self, show: bool = True):
        if not show:
            return
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
    demo = MazeDemo()
    demo.run()
