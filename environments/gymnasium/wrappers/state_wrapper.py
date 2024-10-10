import gymnasium as gym
import numpy as np


class StateWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def get_occluded_states(self) -> np.ndarray:
        return self.env.get_wrapper_attr("get_occluded_states")()

    def get_all_possible_states(self, agent: str | None = None):
        return self.env.get_wrapper_attr("get_all_possible_states")(agent)
