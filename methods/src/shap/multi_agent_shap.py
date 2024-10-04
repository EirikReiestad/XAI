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

    def shap_values(self, test_states: np.ndarray | None = None) -> list[np.ndarray]:
        shap_values = []
        for agent_shap in self.agent_shaps:
            if test_states is not None:
                shap_values.append(agent_shap.shap_values(test_states))
            else:
                shap_values.append(agent_shap.shap_values())
        return shap_values

    def plot(
        self,
        shap_values: Any,
        feature_names: list[str] | None = None,
        include: list[str] | None = None,
        states: np.ndarray | None = None,
        show: bool = True,
        folderpath: str = "",
        filename: str = "",
    ):
        assert len(shap_values) == len(self.agent_shaps)
        plots = []
        for i, (agent_shap, agent_shap_values) in enumerate(
            zip(self.agent_shaps, shap_values)
        ):
            agent_filename = str(i) + filename
            plot = agent_shap.plot(
                agent_shap_values,
                feature_names=feature_names,
                include=include,
                states=states,
                show=show,
                folderpath=folderpath,
                filename=agent_filename,
            )
            plots.append(plot)
        return plots
