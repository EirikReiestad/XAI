from dataclasses import dataclass
from environments.gymnasium.utils import Position
from enum import Enum


class AgentType(Enum):
    """Enum for the type of agent."""

    AGENT0 = 0
    AGENT1 = 1


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
    agent0: Agent
    agent1: Agent
    _active_agent: AgentType

    @property
    def active(self):
        return self.agent0 if self.active_agent == AgentType.AGENT0 else self.agent1

    @property
    def inactive(self):
        return self.agent1 if self.active_agent == AgentType.AGENT0 else self.agent0

    @property
    def colliding(self):
        return self.agent0 == self.agent1

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
            AgentType.AGENT0 if agent == AgentType.AGENT1 else AgentType.AGENT1
        )

        self._active_agent = agent
        self._inactive_agent = inactive_agent

    @property
    def inactive_agent(self) -> AgentType:
        return self._inactive_agent
