import numpy as np
from environments.gymnasium.envs.coop.utils import TileType


class EnvUtils:
    @staticmethod
    def is_within_bounds(env: np.ndarray, x: int | float, y: int | float):
        return 0 <= x < env.shape[0] and 0 <= y < env.shape[1]

    @staticmethod
    def is_not_obstacle(env: np.ndarray, x: int, y: int):
        return env[x, y] != TileType.OBSTACLE.value
