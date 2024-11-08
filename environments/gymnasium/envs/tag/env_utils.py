import numpy as np
from environments.gymnasium.envs.tag.utils import (
    TileType,
    create_object,
    ObjectType,
    Object,
)
from environments.gymnasium.utils import Position


class EnvUtils:
    @staticmethod
    def is_within_bounds(env: np.ndarray, position: Position):
        if env.shape[0] is None:
            raise ValueError("Environment shape is not defined")
        return 0 <= position.x < env.shape[0] and 0 <= position.y < env.shape[1]

    @staticmethod
    def get_object(env: np.ndarray, position: Position) -> Object | None:
        value = env[position.row_major_order]
        if value == TileType.OBSTACLE.value:
            return create_object(ObjectType.OBSTACLE, position)
        if value == TileType.BOX.value:
            return create_object(ObjectType.BOX, position)
        if value == TileType.POWERUP0.value:
            return create_object(ObjectType.POWERUP0, position)
        if value == TileType.POWERUP1.value:
            return create_object(ObjectType.POWERUP1, position)
        return None

    @staticmethod
    def is_obstacle(obj: Object) -> bool:
        if obj.grabable or obj.consumeable:
            return False
        return True
