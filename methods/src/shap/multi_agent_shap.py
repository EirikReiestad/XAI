import random
from itertools import count
from typing import Any

import numpy as np
import shap
import torch

from environments.gymnasium.wrappers import MultiAgentEnv
from .base import MultiAgentBase
import rl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentShap(MultiAgentBase):
    def __init__(self, env: MultiAgentEnv, model: rl.MultiAgentBase):
        self.env = env
        self.models = model.models
        self.model = model
        self.background_states, self.test_states = self._sample_states(10000)

    def explain(self) -> list[np.ndarray]:
        shap_values = [self._explain_single_agent(agent) for agent in self.models]
        return shap_values

    def plot(self, shap_values: Any, **kwargs):
        feature_names = kwargs.get("feature_names", None)
        mean_shap_values = shap_values.mean(axis=2)
        shap.summary_plot(
            mean_shap_values, self.test_states, feature_names=feature_names
        )

    def _explain_single_agent(self, agent: rl.SingleAgentBase) -> Any:
        explainer = shap.Explainer(agent.predict, self.background_states)
        shap_values = explainer(self.test_states).values
        return shap_values

    def _sample_states(self, num_states: int, test: float = 0.2):
        states = self._generate_states(num_states)
        background_states = states[: int(num_states * (1 - test))]
        test_states = states[int(num_states * (1 - test)) :]
        return background_states, test_states

    def _generate_states(self, num_states) -> np.ndarray:
        states: list[np.ndarray] = []
        sample_prob = 0.1

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
