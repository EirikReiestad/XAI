import logging
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from environments.gymnasium.wrappers import MultiAgentEnv, StateWrapper
from methods.src.saliency_map import SaliencyMap
from renderer import Renderer
from rl.src.dqn.wrapper import MultiAgentDQN
from rl.src.common.getter import get_torch_from_numpy
from rl.src.dqn.common.q_values_map import get_q_values_map
from rl.src.managers import WandBConfig

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TagDemo:
    """Demo for the CartPole-v1 environment."""

    def __init__(self) -> None:
        self.episode_information = EpisodeInformation(
            durations=[], rewards=[], object_moved_distance=[]
        )
        env = gym.make("TagEnv-v0", render_mode="rgb_array")
        model_name = "tag-v0"
        self.env = MultiAgentEnv(env)
        wandb_config = WandBConfig(project="tag-v0-local")
        self.dqn = MultiAgentDQN(
            self.env,
            2,
            "dqnpolicy",
            wandb=True,
            wandb_config=wandb_config,
            model_name=model_name,
            save_model=True,
        )

    def run(self):
        self.dqn.learn(100)

        self.plotter = Plotter()
        self.renderer = Renderer(10, 10, 600, 600)
        self.saliency_map = SaliencyMap()

        plt.ion()

        self.env = StateWrapper(self.env)

        for i_episode in range(1000):
            self.dqn.learn(10)
            state, _ = self.env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

            agent_rewards = [0, 0]

            for t in count():
                predicted_actions = self.dqn.predict(state)
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
                ) = self.env.get_wrapper_attr("step_multiple")(actions)

                agent_rewards = [e + r for e, r in zip(agent_rewards, rewards)]

                # self.render_q_values_map(full_state)
                # self.render_saliency_map(observation)

                if terminated or any(terminals) or any(truncated):
                    self.episode_information.durations.append(t + 1)
                    self.episode_information.rewards.append(agent_rewards[0])
                    self.plotter.update(self.episode_information)
                    break
        self.env.close()
        logging.info("Complete")
        self.plotter.update(self.episode_information, show_result=True)
        plt.ioff()
        plt.show()

    def render_saliency_map(self, state: np.ndarray):
        rgb = self.env.render()

        occluded_states = self.env.get_occluded_states()
        torch_state = get_torch_from_numpy(state)
        torch_state.unsqueeze(0)
        saliency_map = self.saliency_map.generate(
            torch_state, occluded_states, self.dqn, agent=0
        )
        if isinstance(rgb, np.ndarray):
            self.renderer.render(background=rgb, saliency_map=saliency_map)

    def render_q_values_map(self, full_state: np.ndarray):
        rgb = self.env.render()

        all_possible_states = self.env.get_all_possible_states()

        states = np.zeros(
            (len(all_possible_states[0]), len(all_possible_states)),
            dtype=torch.Tensor,
        )

        for i, row in enumerate(all_possible_states):
            for j, column in enumerate(row):
                column = np.array(column, dtype=np.float32)
                torch_column = get_torch_from_numpy(column)
                states[j, i] = torch_column

        q_values = self.dqn.get_q_values(states, 0)
        q_values_map = get_q_values_map(states=full_state, q_values=q_values)

        if isinstance(rgb, np.ndarray):
            self.renderer.render(background=rgb, q_values=q_values_map)


if __name__ == "__main__":
    demo = TagDemo()
    demo.run()
