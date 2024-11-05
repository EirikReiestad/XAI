from environments.gymnasium.envs.tag.utils.agent_type import AgentType
from environments.gymnasium.utils import Position

from .utils.agents import Agent, DualAgents


class AgentHandler:
    def __init__(self) -> None:
        self._seeker_slow_factor = 1
        self._hider_slow_factor = 1
        self._freeze_hider = False

    def set_agents(self, seeker_position: Position, hider_position: Position) -> None:
        seeker = Agent(seeker_position)
        hider = Agent(hider_position)
        self.agents = DualAgents(seeker, hider)

    def agent_slow_factor(self, agent: AgentType) -> int:
        if agent == AgentType.SEEKER:
            return self._seeker_slow_factor
        else:
            return self._hider_slow_factor

    def can_move(self) -> bool:
        if self._freeze_hider and self.agents.active_agent == AgentType.HIDER:
            return False
        return True

    @property
    def seeker_slow_factor(self) -> int:
        return self._seeker_slow_factor

    @seeker_slow_factor.setter
    def seeker_slow_factor(self, value: int) -> None:
        self._seeker_slow_factor = value

    @property
    def hider_slow_factor(self) -> int:
        return self._hider_slow_factor

    @hider_slow_factor.setter
    def hider_slow_factor(self, value: int) -> None:
        self._hider_slow_factor = value
