import logging
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from environments.gymnasium.wrappers import MultiAgentEnv
from renderer import Renderer
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
        self.renderer = Renderer(10, 10, 600, 600)
        env = gym.make("TagEnv-v0", render_mode="rgb_array")
        self.env: MultiAgentEnv = MultiAgentEnv(env)

    def run(self):
        dqn = MultiAgentDQN(self.env, 2, "dqnpolicy", wandb=True)

        plt.ion()

        self.env.reset()

        try:
            for i_episode in range(1000):
                dqn.learn(1)

                state, _ = self.env.reset()
                state = torch.tensor(
                    state, device=device, dtype=torch.float32
                ).unsqueeze(0)

                agent_rewards = [0, 0]

                for t in count():
                    predicted_actions = dqn.predict(state)
                    actions = [action.item() for action in predicted_actions]
                    (
                        full_state,
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
                        full_state, device=device, dtype=torch.float32
                    ).unsqueeze(0)

                    rgb = self.env.render()

                    all_possible_states = self.env.get_wrapper_attr(
                        "get_all_possible_states"
                    )

                    q_values = dqn.get_q_values(all_possible_states, 0)
                    print(rgb)
                    if isinstance(rgb, np.ndarray):
                        self.renderer.render(background=rgb, q_values=q_values)

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
