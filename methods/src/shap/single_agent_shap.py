import random
from itertools import count
from typing import Any

import gymnasium as gym
import numpy as np
import shap
import torch
from .base import SingleAgentBase
from .utils import ShapType

import rl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleAgentShap(SingleAgentBase):
    def __init__(
        self,
        env: gym.Env,
        model: rl.SingleAgentBase,
        samples: int,
        shap_type: ShapType = ShapType.IMAGE,
    ):
        self.env = env
        self.model = model
        self.background_states, self.test_states = self._sample_states(samples)
        self.shap_type = shap_type

    def explain(self) -> np.ndarray:
        explainer = shap.Explainer(self.model.predict, self.background_states[0].shape)
        shap_values = explainer(self.test_states).values
        return shap_values

    def plot(
        self,
        shap_values: Any,
        feature_names: list[str] | None = None,
        include: list[str] | None = None,
    ):
        test_states = self.test_states
        if include is not None and feature_names is not None:
            included_indices = [
                i for i, name in enumerate(feature_names) if name in include
            ]
            shap_values = shap_values[:, included_indices, :]
            test_states = test_states[:, included_indices]
            feature_names = include

        mean_shap_values = shap_values.mean(axis=2)
        if self.shap_type == ShapType.BEESWARM:
            shap.summary_plot(
                mean_shap_values, test_states, feature_names=feature_names
            )
        elif self.shap_type == ShapType.IMAGE:
            shap.image_plot(mean_shap_values, test_states, feature_names=feature_names)

    def _sample_states(
        self, num_states: int, test: float = 0.2, sample_prob: float = 0.4
    ):
        states = self._generate_states(num_states, sample_prob)
        background_states = states[: int(num_states * (1 - test))]
        test_states = states[int(num_states * (1 - test)) :]
        return background_states, test_states

    def _generate_states(self, num_states: int, sample_prob: float) -> np.ndarray:
        states: list[np.ndarray] = []

        state, _ = self.env.reset()
        for _ in count():
            state, _ = self.env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

            for _ in count():
                action = self.model.predict_action(state)
                observation, _, terminated, truncated, _ = self.env.step(action.item())
                state = torch.tensor(
                    observation, device=device, dtype=torch.float32
                ).unsqueeze(0)

                if len(states) >= num_states:
                    return np.vstack(states)

                if random.random() < sample_prob:
                    numpy_state = state.cpu().numpy()
                    states.append(numpy_state)

                if terminated or truncated:
                    break

        return np.vstack(states)
