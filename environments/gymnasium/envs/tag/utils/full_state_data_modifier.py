from dataclasses import dataclass
import numpy as np
from environments.gymnasium.envs.tag.utils import (
    FullStateDataExtractor,
    AgentType,
    AGENT_TILE_TYPE,
)
from environments.gymnasium.envs.tag.utils.tile_type import TileType
from environments.gymnasium.utils import Position


@dataclass
class FullStateDataModifier:
    @staticmethod
    def remove_agent(state: np.ndarray, agent: AgentType) -> np.ndarray:
        new_state = state.copy()
        agent_position = FullStateDataExtractor.get_agent_position(state, agent)
        if new_state[agent_position.row_major_order] == TileType.EMPTY.value:
            raise ValueError("Agent already removed")
        new_state[agent_position.row_major_order] = TileType.EMPTY.value
        return new_state

    @staticmethod
    def remove_agents(state: np.ndarray, agents: list[AgentType]) -> np.ndarray:
        new_state = state.copy()
        for agent in agents:
            new_state = FullStateDataModifier.remove_agent(new_state, agent)
        return new_state

    @staticmethod
    def occlude(state: np.ndarray, position: Position) -> np.ndarray:
        state_copy = state.copy()
        state_copy[position.row_major_order] = TileType.EMPTY.value
        return state_copy

    @staticmethod
    def place_agent(
        state: np.ndarray, position: Position, agent: AgentType
    ) -> np.ndarray:
        tile_type = AGENT_TILE_TYPE[agent]
        state_copy = state.copy()
        state_copy[position.row_major_order] = tile_type
        return state_copy
