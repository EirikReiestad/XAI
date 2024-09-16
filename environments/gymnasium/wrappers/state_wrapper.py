import gymnasium as gym
import numpy as np


class StateWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def get_occluded_states(self, state: np.ndarray) -> np.ndarray:
        return self.get_wrapper_attr("get_occlued_states")(state)

    def get_all_possible_states(self):
        return self.env.get_wrapper_attr("get_all_possible_states")()
