import logging
import random
from itertools import count
from typing import Any

import numpy as np
import shap
import torch

import rl
from environments.gymnasium.wrappers import MultiAgentEnv
from .utils import ShapType

from .base import MultiAgentBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentShap(MultiAgentBase):
    def __init__(
        self,
        env: MultiAgentEnv,
        model: rl.MultiAgentBase,
        samples: int,
<<<<<<< HEAD
        shap_type: ShapType,
=======
>>>>>>> 41574cb (Cfeat shap iamge)
    ):
        self.env = env
        self.models = model.models
        self.model = model
        self.background_states, self.test_states = self._sample_states(samples)
        self.shap_type = shap_type

        logging.info("MultiAgentShap initialized")

    def explain(self) -> list[np.ndarray]:
        shap_values = [self._explain_single_agent(agent) for agent in self.models]
        return shap_values

    def plot(
        self,
        shap_values: Any,
        plot_type: ShapType,
        feature_names: list[str] | None = None,
        include: list[str] | None = None,
    ):
        test_states = self.test_states
        if include is not None and feature_names is not None:
            included_indices = [
                i for i, name in enumerate(feature_names) if name in include
            ]
            shap_values = [
                agent_shap_values[:, included_indices, :]
                for agent_shap_values in shap_values
            ]
            test_states = test_states[:, included_indices]
            feature_names = include

        for agent_shap_values in shap_values:
            mean_shap_values = agent_shap_values.mean(axis=2)
            if plot_type == ShapType.BEESWARM:
                shap.summary_plot(
                    mean_shap_values, test_states, feature_names=feature_names
                )
            elif plot_type == ShapType.IMAGE:
                shap.image_plot(
                    mean_shap_values, test_states, feature_names=feature_names
                )

    def _explain_single_agent(self, agent: rl.SingleAgentBase) -> Any:
        explainer = shap.Explainer(agent.predict, self.background_states)
        shap_values = explainer(self.test_states).values
        return shap_values

    def _sample_states(
        self, num_states: int, test: float = 0.2, sample_prob: float = 0.4
    ):
        states = self._generate_states(num_states, sample_prob)
        background_states = states[: int(num_states * (1 - test))]
        test_states = states[int(num_states * (1 - test)) :]
        return background_states, test_states

    def _generate_states(self, num_states: int, sample_prob: float) -> np.ndarray:
        states: list[np.ndarray] = []

        for _ in count():
            state, _ = self.env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

            for _ in count():
                predicted_actions = self.model.predict_actions(state)
                actions = [action.item() for action in predicted_actions]
                (_, observation, terminated, _, _, _, truncated, _) = (
                    self.env.get_wrapper_attr("step_multiple")(actions)
                )

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
