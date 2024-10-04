from typing import Any

import gymnasium as gym
import numpy as np
import shap
import torch

import rl
from environments.gymnasium.wrappers import MultiAgentEnv

from .multi_agent_shap import MultiAgentShap
from .single_agent_shap import SingleAgentShap
from .utils import ShapType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Shap:
    def __init__(
        self,
        env: gym.Env | MultiAgentEnv,
        model: rl.SingleAgentBase | rl.MultiAgentBase,
        samples: int = 100,
        shap_type: ShapType = ShapType.IMAGE,
    ):
        self.multi_agent = isinstance(env, MultiAgentEnv)

        if isinstance(env, MultiAgentEnv) and not isinstance(model, rl.MultiAgentBase):
            raise ValueError("Model must be MultiAgentBase")
        elif not isinstance(env, MultiAgentEnv) and isinstance(
            model, rl.MultiAgentBase
        ):
            raise ValueError("Model must be SingleAgentBase")

        self.env = env
        self.model = model
        self.shap_type = shap_type

        if self.multi_agent:
            assert isinstance(self.env, MultiAgentEnv)
            assert isinstance(self.model, rl.MultiAgentBase)
            self.explainer = MultiAgentShap(self.env, self.model, samples)
        else:
            assert isinstance(self.env, gym.Env)
            assert isinstance(self.model, rl.SingleAgentBase)
            self.explainer = SingleAgentShap(self.env, self.model, samples)

    def explain(
        self,
    ) -> (
        shap.GradientExplainer
        | shap.Explainer
        | list[shap.GradientExplainer | shap.Explainer]
    ):
        return self.explainer.explain()

    def shap_values(self, states: np.ndarray | None = None) -> Any:
        if states is not None:
            return self.explainer.shap_values(states)
        return self.explainer.shap_values()

    def plot(self, shap_values: Any, show: bool = True, **kwargs):
        feature_names = kwargs.get("feature_names", None)
        include = kwargs.get("include", None)
        return self.explainer.plot(
            shap_values,
            feature_names=feature_names,
            include=include,
            show=show,
        )
