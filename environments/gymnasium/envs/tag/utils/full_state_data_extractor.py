import random
import traceback
from dataclasses import dataclass

import numpy as np

from environments.gymnasium.envs.tag.utils import AgentType
from environments.gymnasium.envs.tag.utils.agent_tile_type import AGENT_TILE_TYPE
from environments.gymnasium.envs.tag.utils.tile_type import TileType
from environments.gymnasium.utils import Position


@dataclass
class FullStateDataExtractor:
    @staticmethod
    def get_empty_positions_around_position(state: np.ndarray, position: Position):
        """Only adjacent positions, not diagonals."""
        empty_positions = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_position = Position(position.x + dx, position.y + dy)
            if (
                0 <= new_position.x < state.shape[1]
                and 0 <= new_position.y < state.shape[0]
                and state[new_position.row_major_order] == TileType.EMPTY.value
            ):
                empty_positions.append(new_position)
        return empty_positions

    @staticmethod
    def has_direct_sight(state: np.ndarray) -> tuple[bool, list[Position]]:
        matrix = state.copy()

        seeker_position = FullStateDataExtractor.get_agent_position(
            state, AgentType.SEEKER
        )
        hider_position = FullStateDataExtractor.get_agent_position(
            state, AgentType.HIDER
        )

        x0, y0 = map(int, seeker_position.tuple)
        x1, y1 = map(int, hider_position.tuple)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy

        positions = []

        while (x0, y0) != (x1, y1):
            if matrix[y0][x0] not in [
                TileType.EMPTY.value,
                TileType.SEEKER.value,
                TileType.HIDER.value,
            ]:
                return False, []
            positions.append(Position(x=x0, y=y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return matrix[y1][x1] != 1, positions[1:]

    @staticmethod
    def get_agent_position(state: np.ndarray, agent: AgentType) -> Position:
        agent_tile_type = AGENT_TILE_TYPE.get(agent)
        if agent_tile_type is None:
            raise ValueError(f"Agent type {agent} is not supported.")
        agent_position = np.where(state == agent_tile_type)
        if len(agent_position[0]) > 1 or len(agent_position[1]) > 1:
            raise ValueError(
                f"More than one agent found in the state {state}\n"
                + "".join(traceback.format_stack())
            )
        if len(agent_position[0]) == 0 or len(agent_position[1]) == 0:
            raise ValueError(
                f"No agent found in the state {state}\n"
                + "".join(traceback.format_stack())
            )
        agent_position = Position(x=agent_position[1][0], y=agent_position[0][0])
        return agent_position

    @staticmethod
    def get_object_positions(
        state: np.ndarray, object_type: TileType
    ) -> list[Position]:
        positions = []
        object_positions = np.where(state == object_type.value)
        if len(object_positions) == 0:
            return positions
        if len(object_positions[0]) == 1:
            positions.append(
                Position(x=object_positions[1][0], y=object_positions[0][0])
            )
            return positions
        for y, x in zip(object_positions[0], object_positions[1]):
            positions.append(Position(x, y))
        return positions

    @staticmethod
    def agent_exist(state: np.ndarray, agent: AgentType) -> bool:
        agent_tile_type = AGENT_TILE_TYPE.get(agent)
        if agent_tile_type is None:
            raise ValueError(f"Agent type {agent} is not supported.")
        agent_position = np.where(state == agent_tile_type)
        if len(agent_position[0]) > 1 or len(agent_position[1]) > 1:
            raise ValueError(f"More than one agent found in the state {state}.")
        return len(agent_position[0]) == 1 and len(agent_position[1]) == 1

    @staticmethod
    def get_positions(state: np.ndarray, tile_type: TileType):
        positions = np.where(state == tile_type.value)
        positions = [Position(x=x, y=y) for x, y in zip(positions[1], positions[0])]
        return positions

    @staticmethod
    def get_random_position(state: np.ndarray, tile_type: TileType) -> Position:
        positions = FullStateDataExtractor.get_positions(state, tile_type)
        return random.choice(positions)

    @staticmethod
    def is_empty_tile(state: np.ndarray, position: Position) -> bool:
        return state[position.row_major_order] == TileType.EMPTY.value
