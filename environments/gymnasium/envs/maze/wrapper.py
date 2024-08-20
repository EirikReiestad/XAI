__credits__ = ["Eirik Reiestad"]

import gymnasium as gym
from environments.gymnasium.envs.maze import MazeEnv


class MazeEnvWrapper(gym.Wrapper):
    def __init__(self, env: MazeEnv):
        super().__init__(env)
