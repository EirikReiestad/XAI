import numpy as np
from environments.gymnasium.envs.tag.utils import TileType
from environments.gymnasium.utils import Position


class EnvUtils:
    @staticmethod
    def is_within_bounds(env: np.ndarray, position: Position):
        if env.shape[0] is None:
            raise ValueError("Environment shape is not defined")
        return 0 <= position.x < env.shape[0] and 0 <= position.y < env.shape[1]

    @staticmethod
    def is_object(env: np.ndarray, position: Position):
        return EnvUtils.is_obstacle(env, position) or EnvUtils.is_box(env, position)

    @staticmethod
    def is_obstacle(env: np.ndarray, position: Position):
        return env[position.row_major_order] == TileType.OBSTACLE.value

    @staticmethod
    def is_box(env: np.ndarray, position: Position):
        return env[position.row_major_order] == TileType.BOX.value
