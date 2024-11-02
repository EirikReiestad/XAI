import logging
import os
import torch.nn as nn

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.gymnasium.wrappers import MultiAgentEnv
from managers.src.wandb_manager import WandBConfig
from methods import Shap
from methods.src.cav import CAV
from rl.src.common.getter import get_torch_from_numpy
from rl.src.dqn.common.q_values_map import get_q_values_map
from rl.src.dqn.wrapper import MultiAgentDQN


class ModelHandler:
    def __init__(
        self, model_artifact: str, version_numbers: list[str], shap_samples: int
    ):
        self.num_agents = 2

        env = gym.make("TagEnv-v0", render_mode="rgb_array")
        self.model_name = "tag-v0"
        self.model_artifact = model_artifact
        self.version_numbers = version_numbers
        self.env = MultiAgentEnv(env)
        self.wandb_config = WandBConfig(project="gui")
        self.load_dqn(
            self.wandb_config, self.model_name, model_artifact, version_numbers
        )
        self.shap_samples = shap_samples

        positive_samples = "random"
        negative_samples = "negative_samples"
        self._save_path_cav = "gui/data/cav/"

        self._cav = CAV(self._model, positive_samples, negative_samples)
        self._cavs = self._load_cavs(self._save_path_cav)

        self.tcav_scores = self._tcav_scores(self._cavs)

        self.load_model(shap_samples)

    def _tcav_scores(self, cavs: dict) -> dict:
        for layer, cav in cavs.items():
            tcav_scores = {}
            for key, value in cav.items():
                tcav_scores[key] = self._cav.tcav_score(value)
                return tcav_scores

    def _save_cavs(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        for name, cav in self._cavs.items():
            for key, value in cav.items():
                np.save(f"{path}/{name}_{key}.npy", value)

    def _load_cavs(self, path: str) -> dict:
        cavs = {}
        for name in os.listdir(path):
            cav = {}
            for key in os.listdir(f"{path}/{name}"):
                cav[key] = np.load(f"{path}/{name}/{key}")
                self._cavs[name] = cav
        return cavs

    @property
    def _model(
        self,
    ) -> nn.Module:
        return self.dqn.agents[0].policy_net

    def load_dqn(
        self,
        wandb_config: WandBConfig,
        model_name: str,
        model_artifact: str,
        version_numbers: list[str],
    ):
        self.dqn = MultiAgentDQN(
            self.env,
            self.num_agents,
            "dqnpolicy",
            wandb_active=False,
            wandb_config=wandb_config,
            model_name=model_name,
            save_every_n_episodes=100,
            save_model=False,
            load_model=True,
            gif=False,
            run_path="eirikreiestad-ntnu/tag-v0-eirik",
            model_artifact=model_artifact,
            version_numbers=version_numbers,
        )

    def update_model(
        self,
        model_artifact: str,
        version_numbers: list[str],
    ):
        if (
            self.model_artifact != model_artifact
            or self.version_numbers != version_numbers
        ) and model_artifact != "":
            self.model_artifact = model_artifact
            self.version_numbers = version_numbers
            logging.info("Loading model")
            self.load_dqn(
                self.wandb_config, self.model_name, model_artifact, version_numbers
            )

    def update_shap(self, samples: int):
        if self.shap_samples != samples:
            self.shap_samples = samples
            logging.info("Loading SHAP")
            self.load_model(samples)

    def load_model(self, samples=10):
        self.shap = Shap(self.env, self.dqn, samples=samples)

    def generate_shap(
        self, states: np.ndarray | None = None, filename: str = "shap.png"
    ):
        self.shap.explain()
        if states is None:
            shap_values = self.shap.shap_values()
        else:
            shap_values = self.shap.shap_values(states)
        self.shap.plot(
            shap_values,
            states=states,
            show=False,
            folderpath="gui/src/assets/",
            filename=filename,
        )

    def generate_q_values(self, states: np.ndarray):
        for _, state in enumerate(states):
            q_values_maps = self.get_q_values_maps(state)
            for i, q_values_map in enumerate(q_values_maps):
                self.generate_saliency_image(
                    matrix=q_values_map, filename=f"gui/src/assets/saliency_{i}.png"
                )

    def get_q_values_maps(self, full_state: np.ndarray) -> list[np.ndarray]:
        self.env.unwrapped.set_state(full_state)
        all_possible_seeker_states = self.env.unwrapped.get_all_possible_states(
            "seeker"
        )
        all_possible_hider_states = self.env.unwrapped.get_all_possible_states("hider")

        return [
            self.get_agent_q_values_map(full_state, all_possible_seeker_states),
            self.get_agent_q_values_map(full_state, all_possible_hider_states),
        ]

    def get_agent_q_values_map(
        self, full_state: np.ndarray, all_possible_states: np.ndarray
    ):
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

    def generate_saliency_image(
        self, matrix: np.ndarray, filename: str = "gui/src/assets/saliency.png"
    ):
        matrix = np.flipud(matrix)
        matrix = np.rot90(matrix)
        matrix = np.rot90(matrix)
        matrix = np.rot90(matrix)
        plt.imshow(matrix, cmap="hot")
        plt.colorbar()
        plt.savefig(filename)
        plt.close()
