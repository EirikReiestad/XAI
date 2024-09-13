import logging
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
from environments.gymnasium.wrappers import MultiAgentEnv

from demo import settings
from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from rl.src.dqn.wrapper import MultiAgentDQN

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TagDemo:
    """Demo for the CartPole-v1 environment."""

    def __init__(self) -> None:
        self.episode_information = EpisodeInformation(
            durations=[], rewards=[], object_moved_distance=[]
        )
        self.plotter = Plotter()
        env = gym.make("TagEnv-v0", render_mode="human")
        self.env: MultiAgentEnv = MultiAgentEnv(env)

    def run(self):
        dqn = MultiAgentDQN(self.env, 2, "dqnpolicy", wandb=False)
        dqn.learn(settings.EPOCHS)
        print("Finished training")

        env = gym.make("TagEnv-v0", render_mode="human")
        self.env: MultiAgentEnv = MultiAgentEnv(self.env)

        plt.ion()

        self.env.reset()

        try:
            for i_episode in range(1000):
                state, _ = self.env.reset()
                state = torch.tensor(
                    state, device=device, dtype=torch.float32
                ).unsqueeze(0)

                agent_rewards = [0, 0]

                for t in count():
                    predicted_actions = dqn.predict(state)
                    actions = [action.item() for action in predicted_actions]
                    (
                        observation,
                        terminated,
                        observations,
                        rewards,
                        terminals,
                        truncated,
                        _,
                    ) = self.env.step_multiple(actions)

                    agent_rewards += rewards

                    state = torch.tensor(
                        observation, device=device, dtype=torch.float32
                    ).unsqueeze(0)
                    self.env.render()

                    if terminated:
                        self.episode_information.durations.append(t + 1)
                        self.episode_information.rewards.append(agent_rewards[0])
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
    demo = TagDemo()
    demo.run()
