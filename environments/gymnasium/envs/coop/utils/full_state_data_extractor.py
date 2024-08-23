from dataclasses import dataclass
import numpy as np
from environments.gymnasium.envs.coop.utils import AgentType
from environments.gymnasium.envs.coop.utils.tile_type import TileType
from environments.gymnasium.envs.coop.utils.agent_tile_type import AGENT_TILE_TYPE
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
            raise ValueError("More than one agent found in the state.")
        if len(agent_position[0]) == 0 or len(agent_position[1]) == 0:
            raise ValueError("No agent found in the state.")
        agent_position = Position(x=agent_position[1][0], y=agent_position[0][0])
        return agent_position

    @staticmethod
    def agent_exist(state: np.ndarray, agent: AgentType) -> bool:
        agent_tile_type = AGENT_TILE_TYPE.get(agent)
        if agent_tile_type is None:
            raise ValueError(f"Agent type {agent} is not supported.")
        agent_position = np.where(state == agent_tile_type)
        if len(agent_position[0]) > 1 or len(agent_position[1]) > 1:
            raise ValueError("More than one agent found in the state.")
        return len(agent_position[0]) == 1 and len(agent_position[1]) == 1

    @staticmethod
    def get_obstacle_positions(state: np.ndarray) -> list[Position]:
        obstacle_positions = np.where(state == TileType.OBSTACLE.value)
        return [
            Position(x=x, y=y)
            for x, y in zip(obstacle_positions[1], obstacle_positions[0])
        ]
