import numpy as np
from environments.gymnasium.envs.coop.utils import TileType
from environments.gymnasium.utils import Position


class EnvUtils:
    @staticmethod
    def is_within_bounds(env: np.ndarray, position: Position):
        return 0 <= position.x < env.shape[0] and 0 <= position.y < env.shape[1]

    @staticmethod
    def is_not_obstacle(env: np.ndarray, position: Position):
        return env[position.row_major_order] != TileType.OBSTACLE.value
