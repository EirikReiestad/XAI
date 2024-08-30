from dataclasses import dataclass
import numpy as np
from environments.gymnasium.envs.maze.utils import TileType, FullStateDataExtractor
from environments.gymnasium.utils import Position


@dataclass
class FullStateDataModifier:
    @staticmethod
    def remove_agent(state: np.ndarray) -> np.ndarray:
        new_state = state.copy()
        agent_position = FullStateDataExtractor.get_agent_position(state)
        if new_state[agent_position.row_major_order] == TileType.EMPTY.value:
            raise ValueError("Agent already removed")
        new_state[agent_position.row_major_order] = TileType.EMPTY.value
        return new_state

    @staticmethod
    def place_agent(state: np.ndarray, position: Position) -> np.ndarray:
        new_state = state.copy()
        new_state[position.row_major_order] = TileType.START.value
        return new_state
