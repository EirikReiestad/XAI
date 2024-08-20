__credits__ = ["Eirik Reiestad"]

import gymnasium as gym

from environments.gymnasium.envs.coop import CoopEnv


class CoopEnvWrapper(gym.Wrapper):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, env: CoopEnv, render_mode: str = "human"):
        super().__init__(env)
