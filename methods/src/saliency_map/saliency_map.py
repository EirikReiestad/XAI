from rl.src.base import SingleAgentBase, MultiAgentBase
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SaliencyMap:
    methods = ["occlusion"]

    def __init__(self, method: str = "occlusion"):
        self._method = method

    def generate(
        self,
        state: torch.Tensor,
        occluded_states: np.ndarray,
        model: SingleAgentBase | MultiAgentBase,
        agent: int = 0,
    ):
        if isinstance(model, SingleAgentBase):
            self.generate_single_agent(state, occluded_states, model)
        elif isinstance(model, MultiAgentBase):
            self.generate_multi_agent(state, occluded_states, model, agent)

    def generate_single_agent(
        self, state: torch.Tensor, occluded_states: np.ndarray, model: SingleAgentBase
    ) -> np.ndarray:
        current_reward = model.predict(state)
        heatmap = np.zeros_like(state)

        for row in range(heatmap.shape[0]):
            for col in range(heatmap.shape[1]):
                occluded_state = state.clone()
                occluded_reward = model.predict(occluded_state)
                heatmap[row, col] = current_reward - occluded_reward

        for i, occluded_state in enumerate(occluded_states):
            occluded_state = torch.tensor(
                occluded_state, device=device, dtype=torch.float32
            ).unsqueeze(0)
            occluded_reward = model.predict(occluded_state)
            heatmap[i] = current_reward - occluded_reward

        return heatmap

    def generate_multi_agent(
        self,
        state: torch.Tensor,
        occluded_states: np.ndarray,
        model: MultiAgentBase,
        agent: int,
    ):
        print(state)
        current_reward = model.predict(state)[agent]
        heatmap = np.zeros_like(state)

        for row in range(heatmap.shape[0]):
            for col in range(heatmap.shape[1]):
                occluded_state = state.clone()
                occluded_reward = model.predict(occluded_state)[agent]
                heatmap[row, col] = current_reward - occluded_reward

        for i, occluded_state in enumerate(occluded_states):
            occluded_state = torch.tensor(
                occluded_state, device=device, dtype=torch.float32
            ).unsqueeze(0)
            occluded_reward = model.predict(occluded_state)[agent]
            heatmap[i] = current_reward - occluded_reward
        return heatmap

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        if method not in self.methods:
            raise ValueError(
                f"Method {method} not supported. Choose from {self.methods}"
            )
        self._method = method
