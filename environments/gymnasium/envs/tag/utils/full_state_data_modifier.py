from dataclasses import dataclass

import numpy as np

from environments.gymnasium.envs.tag.utils import (
    AGENT_TILE_TYPE,
    AgentType,
    FullStateDataExtractor,
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

    @staticmethod
    def random_agent_position(state: np.ndarray, agent: AgentType) -> np.ndarray:
        removed_agent_state = FullStateDataModifier.remove_agent(state, agent)
        random_position = FullStateDataExtractor.get_random_position(
            removed_agent_state, TileType.EMPTY
        )
        return FullStateDataModifier.place_agent(
            removed_agent_state, random_position, agent
        )

    @staticmethod
    def remove_objects(
        state: np.ndarray, object_type: TileType
    ) -> tuple[np.ndarray, int]:
        new_state = state.copy()
        object_positions = FullStateDataExtractor.get_object_positions(
            state, object_type
        )
        for object_position in object_positions:
            if new_state[object_position.row_major_order] == TileType.EMPTY.value:
                raise ValueError("Object already removed")
            new_state[object_position.row_major_order] = TileType.EMPTY.value
        return new_state, len(object_positions)

    @staticmethod
    def random_objects_position(state: np.ndarray, object_type: TileType) -> np.ndarray:
        removed_objects_state, num_objects_removed = (
            FullStateDataModifier.remove_objects(state, object_type)
        )
        state_copy = state.copy()
        for _ in range(num_objects_removed):
            random_position = FullStateDataExtractor.get_random_position(
                removed_objects_state, TileType.EMPTY
            )
            removed_objects_state[random_position.row_major_order] = object_type.value
        state_copy = removed_objects_state
        return state_copy
