from dataclasses import dataclass
from environments.gymnasium.utils import Position
from enum import Enum


class AgentType(Enum):
    """Enum for the type of agent."""

    SEEKER = 0
    HIDER = 1


@dataclass
class Agent:
    _position: Position

    def __eq__(self, other):
        return self.position == other.position

    @property
    def position(self) -> Position:
        return self._position

    @position.setter
    def position(self, position: Position):
        self._position = position


@dataclass
class DualAgents:
    seeker: Agent
    hider: Agent
    _active_agent: AgentType

    @property
    def active(self):
        return self.seeker if self.active_agent == AgentType.SEEKER else self.hider

    @property
    def inactive(self):
        return self.hider if self.active_agent == AgentType.SEEKER else self.seeker

    @property
    def colliding(self):
        return self.seeker == self.hider

    @property
    def active_agent(self) -> AgentType:
        return self._active_agent

    @active_agent.setter
    def active_agent(self, agent: AgentType | int):
        if isinstance(agent, int):
            agent = AgentType(agent)
        if agent not in AgentType:
            raise ValueError(f"Invalid agent type {agent}")

        inactive_agent = (
            AgentType.SEEKER if agent == AgentType.HIDER else AgentType.HIDER
        )

        self._active_agent = agent
        self._inactive_agent = inactive_agent

    @property
    def inactive_agent(self) -> AgentType:
        return self._inactive_agent
