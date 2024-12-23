from environments.gymnasium.envs.tag.utils.agent_type import AgentType
from environments.gymnasium.envs.tag.utils import ObjectType
from environments.gymnasium.utils import Position

from .utils.agents import Agent, DualAgents


class AgentController:
    def __init__(self, slow_factor: int) -> None:
        self._freeze_hider = False
        self._init_slow_factor = slow_factor
        self._slow_factor = self._init_slow_factor

    def reset(self) -> None:
        self._slow_factor = self._init_slow_factor

    def add_slow_factor(self, value: int) -> None:
        self._slow_factor += value

    def decrease_slow_factor(self, value: int) -> None:
        self._slow_factor -= value
        self._slow_factor = max(1, self._slow_factor)

    @property
    def slow_factor(self) -> int:
        return self._slow_factor

    @slow_factor.setter
    def slow_factor(self, value: int) -> None:
        self._slow_factor = value


class AgentHandler:
    def __init__(self) -> None:
        self._freeze_hider = False
        self._seeker_controller = AgentController(1)
        self._hider_controller = AgentController(2)

    def reset(self) -> None:
        self._seeker_controller.reset()
        self._hider_controller.reset()

    def set_agents(self, seeker_position: Position, hider_position: Position) -> None:
        seeker = Agent(seeker_position)
        hider = Agent(hider_position)
        self.agents = DualAgents(seeker, hider)

    def agent_slow_factor(self, agent: AgentType) -> int:
        if agent == AgentType.SEEKER:
            return self._seeker_controller.slow_factor
        else:
            return self._hider_controller.slow_factor

    def can_move(self, steps: int) -> bool:
        if self.agents.active_agent == AgentType.SEEKER:
            return self._can_move_seeker(steps)
        return self._can_move_hider(steps)

    def move_in_object(self, object_type: ObjectType) -> None:
        if object_type == ObjectType.POWERUP0:
            if self.agents.active_agent == AgentType.SEEKER:
                self._seeker_controller.decrease_slow_factor(1)
            else:
                self._hider_controller.decrease_slow_factor(1)
        if object_type == ObjectType.POWERUP1:
            if self.agents.active_agent == AgentType.SEEKER:
                self._hider_controller.add_slow_factor(1)
                self._normalize_slow_factor()
            else:
                self._seeker_controller.add_slow_factor(1)
                self._normalize_slow_factor()

    def _normalize_slow_factor(self) -> None:
        if self._seeker_controller.slow_factor < self._hider_controller.slow_factor:
            scale = 1 / self._seeker_controller.slow_factor
            self._seeker_controller.slow_factor = 1
            self._hider_controller.slow_factor = int(
                self._hider_controller.slow_factor * scale
            )
        else:
            scale = 1 / self._hider_controller.slow_factor
            self._hider_controller.slow_factor = 1
            self._seeker_controller.slow_factor = int(
                self._seeker_controller.slow_factor * scale
            )

    def set_agent_slow_factor(self, agent: AgentType, value: int) -> None:
        if agent == AgentType.SEEKER:
            self._seeker_controller.slow_factor = value
        else:
            self._hider_controller.slow_factor = value

    def _can_move_hider(self, steps: int) -> bool:
        return False
        if steps % self._hider_controller.slow_factor == 0:
            return not self._freeze_hider
        return False

    def _can_move_seeker(self, steps: int) -> bool:
        if steps % self._seeker_controller.slow_factor == 0:
            return True
        return False
