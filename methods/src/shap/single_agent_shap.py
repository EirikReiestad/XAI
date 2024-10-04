from typing import Any

import gymnasium as gym
import numpy as np
import shap
import torch
from .base import SingleAgentBase
from .utils import ShapType, sample_states

import rl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleAgentShap(SingleAgentBase):
    def __init__(
        self,
        env: gym.Env,
        model: rl.SingleAgentBase,
        samples: int,
        shap_type: ShapType = ShapType.IMAGE,
        states: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.env = env
        self.model = model
        if states is None:
            self.background_states, self.test_states = sample_states(
                env, model, samples
            )
        else:
            self.background_states, self.test_states = states
        self.shap_type = shap_type

    def explain(self) -> shap.GradientExplainer | shap.Explainer:
        if self.shap_type == ShapType.IMAGE:
            states = torch.tensor(self.background_states, device=device)
            self.explainer = shap.GradientExplainer(self.model.policy_net, states)
        elif self.shap_type == ShapType.BEESWARM:
            self.explainer = shap.Explainer(self.model.predict, self.background_states)
        else:
            raise ValueError(f"Invalid shap type: {self.shap_type}")
        return self.explainer

    def shap_values(self, test_states: np.ndarray | None = None) -> Any:
        states = self.test_states
        if test_states is not None:
            states = test_states
        if self.shap_type == ShapType.IMAGE:
            input_states = torch.tensor(states, device=device)
        else:
            input_states = states
        shap_values = self.explainer(input_states).values
        return shap_values

    def plot(
        self,
        shap_values: Any,
        feature_names: list[str] | None = None,
        include: list[str] | None = None,
        states: np.ndarray | None = None,
        show: bool = True,
    ):
        test_states = self.test_states
        if states is not None:
            test_states = states

        if include is not None and feature_names is not None:
            included_indices = [
                i for i, name in enumerate(feature_names) if name in include
            ]
            shap_values = shap_values[:, included_indices, :]
            test_states = test_states[:, included_indices]
            feature_names = include

        if self.shap_type == ShapType.BEESWARM:
            mean_shap_values = shap_values.mean(axis=2)
            return shap.summary_plot(
                mean_shap_values, test_states, feature_names=feature_names
            )
        elif self.shap_type == ShapType.IMAGE:
            return shap.image_plot(shap_values, test_states, show=show)
