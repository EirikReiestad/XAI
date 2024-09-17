import numpy as np
from environments.gymnasium.envs.maze.utils import TileType
from environments.gymnasium.utils import Position


class MazeUtils:
    @staticmethod
    def is_within_bounds(env: np.ndarray, position: Position):
        return 0 <= position.x < env.shape[0] and 0 <= position.y < env.shape[1]

    @staticmethod
    def is_not_obstacle(env: np.ndarray, position: Position):
        return env[position.row_major_order] != TileType.OBSTACLE.value

    @staticmethod
    def validate_maze(env: np.ndarray, height: int, width: int):
        """Validates the maze file's content."""
        if not env:
            raise ValueError("No data: Failed to read maze from file.")
        if len(env) != height or len(env[0]) != width:
            raise ValueError(
                f"Invalid maze size. Expected {height}x{width}, got {len(env)}x{len(env[0])}"
            )
        flatten_maze = np.array(env)
        if np.all(flatten_maze != TileType.END.value):
            raise ValueError("The goal position is not set.")
        if np.all(flatten_maze != TileType.START.value):
            raise ValueError("The start position is not set.")

    @staticmethod
    def is_not_goal(env: np.ndarray, position: Position):
        return env[position.row_major_order] != TileType.END.value
