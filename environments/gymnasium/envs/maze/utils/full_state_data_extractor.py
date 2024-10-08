from dataclasses import dataclass
import numpy as np
from environments.gymnasium.envs.maze.utils import TileType
from environments.gymnasium.utils import Position


@dataclass
class FullStateDataExtractor:
    @staticmethod
    def get_agent_position(state: np.ndarray) -> Position:
        agent_position = np.where(state == TileType.START.value)
        if len(agent_position[0]) > 1 or len(agent_position[1]) > 1:
            raise ValueError("More than one agent found in the state.")
        if len(agent_position[0]) == 0 or len(agent_position[1]) == 0:
            raise ValueError("No agent found in the state.")
        agent_position = Position(x=agent_position[1][0], y=agent_position[0][0])
        return agent_position

    @staticmethod
    def get_goal_position(state: np.ndarray) -> Position:
        goal_position = np.where(state == TileType.END.value)
        if len(goal_position[0]) > 1 or len(goal_position[1]) > 1:
            raise ValueError("More than one goal found in the state.")
        if len(goal_position[0]) == 0 or len(goal_position[1]) == 0:
            raise ValueError("No goal found in the state.")
        goal_position = Position(x=goal_position[1][0], y=goal_position[0][0])
        return goal_position

    @staticmethod
    def get_obstacle_positions(state: np.ndarray) -> list[Position]:
        obstacle_positions = np.where(state == TileType.OBSTACLE.value)
        obstacle_positions = [
            Position(x=x, y=y)
            for x, y in zip(obstacle_positions[1], obstacle_positions[0])
        ]
        return obstacle_positions

    @staticmethod
    def is_empty_tile(state: np.ndarray, position: Position) -> bool:
        return state[position.row_major_order] == TileType.EMPTY.value
