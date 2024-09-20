import random
from itertools import count
from typing import Any

import gymnasium as gym
import numpy as np
import shap
import torch
from .base import SingleAgentBase

import rl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleAgentShap(SingleAgentBase):
    def __init__(self, env: gym.Env, model: rl.SingleAgentBase):
        self.env = env
        self.model = model
        self.background_states, self.test_states = self._sample_states(10000)

    def explain(self) -> np.ndarray:
        explainer = shap.Explainer(self.model.predict, self.background_states)
        shap_values = explainer(self.test_states).values
        return shap_values

    def plot(self, shap_values: Any, **kwargs):
        feature_names = kwargs.get("feature_names", None)
        mean_shap_values = shap_values.mean(axis=2)
        shap.summary_plot(
            mean_shap_values, self.test_states, feature_names=feature_names
        )

    def _sample_states(self, num_states: int, test: float = 0.2):
        states = self._generate_states(num_states)
        background_states = states[: int(num_states * (1 - test))]
        test_states = states[int(num_states * (1 - test)) :]
        return background_states, test_states

    def _generate_states(self, num_states) -> np.ndarray:
        states: list[np.ndarray] = []
        sample_prob = 0.1

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
