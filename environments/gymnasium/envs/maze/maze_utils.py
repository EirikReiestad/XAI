import numpy as np
from environments.gymnasium.envs.maze.utils import TileType


class MazeUtils:
    @staticmethod
    def is_within_bounds(env: np.ndarray, x: int | float, y: int | float):
        return 0 <= x < env.shape[0] and 0 <= y < env.shape[1]

    @staticmethod
    def is_not_obstacle(env: np.ndarray, x: int, y: int):
        return env[x, y] != TileType.OBSTACLE.value

    @staticmethod
    def validate_maze(env: np.ndarray, height: int, width: int):
        """Validates the maze file's content."""
        if not env:
            raise ValueError("No data: Failed to read maze from file.")
        if len(env) != height or len(env[0]) != width:
            raise ValueError(
                f"Invalid maze size. Expected {height}x{width}, got {len(env)}x{len(env[0])}"
            )
        flatten_maze = np.array(env).flatten()
        if np.all(flatten_maze != TileType.END.value):
            raise ValueError("The goal position is not set.")
        if np.all(flatten_maze != TileType.START.value):
            raise ValueError("The start position is not set.")
