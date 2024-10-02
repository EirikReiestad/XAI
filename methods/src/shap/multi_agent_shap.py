from typing import Any

import numpy as np
import shap
import torch

import rl
from environments.gymnasium.wrappers import MultiAgentEnv
from .utils import ShapType, sample_states

from .base import MultiAgentBase

from .single_agent_shap import SingleAgentShap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentShap(MultiAgentBase):
    def __init__(
        self,
        env: MultiAgentEnv,
        model: rl.MultiAgentBase,
        samples: int,
        shap_type: ShapType = ShapType.IMAGE,
    ):
        self.background_states, self.test_states = sample_states(env, model, samples)
        self.agent_shaps = []
        for model in model.models:
            if not isinstance(model, rl.SingleAgentBase):
                raise ValueError("Model is unknown")
            self.agent_shaps.append(
                SingleAgentShap(
                    env,
                    model,
                    samples,
                    shap_type,
                )
            )

    def explain(self) -> list[shap.GradientExplainer | shap.Explainer]:
        explainers = []
        for agent_shap in self.agent_shaps:
            explainer = agent_shap.explain()
            explainers.append(explainer)
        return explainers

    def shap_values(self) -> list[np.ndarray]:
        shap_values = []
        for agent_shap in self.agent_shaps:
            shap_values.append(agent_shap.shap_values())
        return shap_values

    def plot(
        self,
        shap_values: Any,
        feature_names: list[str] | None = None,
        include: list[str] | None = None,
        states: np.ndarray | None = None,
    ):
        for agent_shap, agent_shap_values in zip(self.agent_shaps, shap_values):
            agent_shap.plot(
                agent_shap_values,
                feature_names=feature_names,
                include=include,
                states=states,
            )
