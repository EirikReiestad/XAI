import getpass
import logging
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from environments.gymnasium.wrappers import MetadataWrapper, MultiAgentEnv, StateWrapper
from methods import SaliencyMap, Shap
from renderer import Renderer
from rl.src.common.getter import get_torch_from_numpy
from rl.src.dqn.common.q_values_map import get_q_values_map
from rl.src.dqn.wrapper import MultiAgentDQN
from managers import WandBConfig

is_ipython = "inline" in matplotlib.get_backend()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TagDemo:
    """Demo for the Tag-v0 environment."""

    def __init__(self) -> None:
        self.episode_information = EpisodeInformation(
            durations=[], rewards=[], object_moved_distance=[]
        )
        self.num_agents = 2

        current_user = getpass.getuser()
        project = f"tag-v0-{current_user}"

        render_mode = "human" if not current_user == "eirre" else "rgb_array"
        env = gym.make("TagEnv-v0", render_mode=render_mode)
        model_name = "tag-v0"
        self.env = MultiAgentEnv(env)

        metadata = MetadataWrapper(self.env)

        wandb_config = WandBConfig(project=project, other=metadata.get_config())
        self.dqn = MultiAgentDQN(
            self.env,
            self.num_agents,
            "dqnpolicy",
            wandb_active=True,
            wandb_config=wandb_config,
            model_name=model_name,
            save_every_n_episodes=100,
            save_model=True,
            load_model=False,
            gif=True,
            run_path="eirikreiestad-ntnu/tag-v0-eirre",
            model_artifact="model_8500",
            version_numbers=["v22", "v23"],
        )

    def run(
        self,
        episodes: int = 1000000,
        num_sweep_episodes: int = 2500,
        sweep: bool = False,
        shap: bool = False,
        show: bool = False,
    ):
        logging.info("Learning...")
        if sweep:
            self.dqn.sweep(num_sweep_episodes)
        else:
            self.dqn.learn(episodes)
        self.shap(shap)
        self.show(show)

    def show(self, run: bool = True):
        if not run:
            return

        plt.ion()

        self.renderer = Renderer(10, 10, 600, 600)
        self.saliency_map = SaliencyMap()

        self.plotter = Plotter(plot_agent=True)
        self.env = StateWrapper(self.env)

        for i_episode in count():
            self.dqn.learn(1)
            state, _ = self.env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

            agent_rewards = [0, 0]

            for t in count():
                predicted_actions = self.dqn.predict_actions(state)
                actions = [action.item() for action in predicted_actions]
                (
                    full_state,
                    observation,
                    terminated,
                    info,
                    observations,
                    rewards,
                    terminals,
                    truncated,
                    _,
                ) = self.env.get_wrapper_attr("step_multiple")(actions)

                agent_rewards = [e + r for e, r in zip(agent_rewards, rewards)]

                self.render(full_state, render="q_values")

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

    def render(self, state: np.ndarray, render: str = ""):
        rgb = self.env.render()

        q_values_map = self.get_q_values_map(state) if render == "q_values" else None
        saliency_map = self._get_saliency_map(state) if render == "saliency" else None

        if isinstance(rgb, np.ndarray):
            self.renderer.render(
                background=rgb, q_values=q_values_map, saliency_map=saliency_map
            )

    def _get_saliency_map(self, state: np.ndarray) -> np.ndarray:
        occluded_states = self.env.get_occluded_states()
        torch_state = get_torch_from_numpy(state)
        torch_state.unsqueeze(0)
        saliency_map = self.saliency_map.generate(
            torch_state, occluded_states, self.dqn, agent=0
        )
        return saliency_map

    def get_q_values_map(self, full_state: np.ndarray):
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
        q_values_map = get_q_values_map(
            states=full_state, q_values=q_values, max_q_values=True
        )
        return q_values_map

    def shap(self, run: bool = True):
        if not run:
            return

        logging.info("Shap setup...")
        shap = Shap(self.env, self.dqn, samples=10)
        logging.info("Explaining...")
        shap.explain()
        shap_values = shap.shap_values()
        env = MetadataWrapper(self.env)
        feature_names = env.feature_names()
        include = (
            [
                "Hider X",
                "Hider Y",
                "Seeker X",
                "Seeker Y",
                "Distance",
                "Direction X",
                "Direction Y",
                "Box 0 x",
                "Box 0 y",
                "Box 0 grabbable",
                "Box 0 grabbed",
            ],
        )
        shap.plot(
            shap_values,
            feature_names=feature_names,
        )


if __name__ == "__main__":
    demo = TagDemo()
    demo.run()
