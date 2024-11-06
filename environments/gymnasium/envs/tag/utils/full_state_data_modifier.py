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
    def place_agents_far_apart(state: np.ndarray, radius: float) -> np.ndarray:
        new_state = state.copy()
        removed_agents_state = FullStateDataModifier.remove_agents(
            new_state, [AgentType.HIDER, AgentType.SEEKER]
        )
        empty_positions = FullStateDataExtractor.get_empty_positions(
            removed_agents_state
        )

        np.random.shuffle(np.array(empty_positions))

        for i, hider_position in enumerate(empty_positions):
            for seeker_position in empty_positions[i + 1 :]:
                if (
                    np.linalg.norm(
                        np.array(hider_position.tuple) - np.array(seeker_position.tuple)
                    )
                    >= radius
                ):
                    seeker_state = FullStateDataModifier.place_agent(
                        new_state, seeker_position, AgentType.SEEKER
                    )
                    hidder_state = FullStateDataModifier.place_agent(
                        seeker_state, hider_position, AgentType.HIDER
                    )
                    return hidder_state
        raise ValueError("Could not place agents far apart")

    @staticmethod
    def place_seeker_next_to_hider(state: np.ndarray) -> np.ndarray:
        new_state = state.copy()
        hider_position = FullStateDataExtractor.get_agent_position(
            state, AgentType.HIDER
        )
        empty_positions_around_position = (
            FullStateDataExtractor.get_empty_positions_around_position(
                state, hider_position
            )
        )
        removed_seeker_position = FullStateDataModifier.remove_agent(
            new_state, AgentType.SEEKER
        )
        if len(empty_positions_around_position) == 0:
            new_state = FullStateDataModifier.random_agent_position(
                new_state, AgentType.HIDER
            )
            return FullStateDataModifier.place_seeker_next_to_hider(new_state)
        random_position = np.random.choice(np.array(empty_positions_around_position))
        placed_seeker = FullStateDataModifier.place_agent(
            removed_seeker_position, random_position, AgentType.SEEKER
        )
        return placed_seeker

    @staticmethod
    def place_agent_next_to_box(state: np.ndarray, agent: AgentType) -> np.ndarray:
        new_state = state.copy()
        removed_agent_state = FullStateDataModifier.remove_agent(new_state, agent)

        box_positions = FullStateDataExtractor.get_object_positions(
            removed_agent_state, TileType.BOX
        )
        box_position = np.random.choice(np.array(box_positions))

        empty_positions_around_position = (
            FullStateDataExtractor.get_empty_positions_around_position(
                state, box_position
            )
        )

        if len(empty_positions_around_position) == 0:
            new_state = FullStateDataModifier.random_objects_position(
                new_state, TileType.BOX
            )
            return FullStateDataModifier.place_agent_next_to_box(new_state, agent)

        random_position = np.random.choice(empty_positions_around_position)
        placed_agent_state = FullStateDataModifier.place_agent(
            removed_agent_state, random_position, agent
        )
        return placed_agent_state

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
