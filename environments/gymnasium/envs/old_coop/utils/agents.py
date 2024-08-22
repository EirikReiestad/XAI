from dataclasses import dataclass
from environments.gymnasium.utils import Position
from enum import Enum


class AgentType(Enum):
    """Enum for the type of agent."""

    AGENT0 = 0
    AGENT1 = 1


@dataclass
class Agent:
    position: Position

    def __eq__(self, other):
        return self.position == other.position


@dataclass
class Agents:
    agent0: Agent
    agent1: Agent
    active_agent: AgentType

    @property
    def active(self):
        return self.agent0 if self.active_agent == AgentType.AGENT0 else self.agent1
