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
