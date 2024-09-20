import random
from itertools import count
from typing import Any

import gymnasium as gym
import numpy as np
import shap
import torch

from rl import SingleAgentBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Shap:
    def __init__(self, env: gym.Env, model: SingleAgentBase):
        self.env = env
        self.model = model
        self.background_states, self.test_states = self._sample_states(1000)

    def explain(self):
        explainer = shap.Explainer(self.model.predict, self.background_states)
        shap_values = explainer(self.test_states)
        return shap_values

    def plot(self, shap_values: Any):
        shap.summary_plot(shap_values, self.test_states)

    def _sample_states(self, num_states: int, test: float = 0.2):
        states = self._generate_states(num_states)
        background_states = states[: int(num_states * test)]
        test_states = states[int(num_states * test) :]
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
