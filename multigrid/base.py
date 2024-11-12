from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable

import gymnasium as gym
from gymnasium import spaces

from .core.agent import Agent, AgentState
from .core.constants import TILE_PIXELS
from .core.grid import Grid

AgentID = int
ObservationType = Dict[str, Any]


class MultiGridEnv(gym.Env, ABC):
    def __init__(
        self,
        agents: Iterable[Agent] | int = 1,
        width: int = 8,
        height: int = 8,
        max_steps: int = 100,
        title_size: int = TILE_PIXELS,
    ):
        gym.Env.__init__(self)

        self._width, self._height = width, height
        self._grid: Grid = Grid(width, height)

        self._max_steps = max_steps
        self._title_size = title_size

        self._agents = agents
        if isinstance(agents, int):
            self._num_agents = agents
            self._agent_states = AgentState(agents)
            self._agents: list[Agent] = []
            for i in range(agents):
                agent = Agent(i)
                agent.state = self._agent_states[i]
                self._agents.append(agent)
        else:
            raise NotImplementedError("Only support integer number of agents")

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> tuple[dict[AgentID, ObservationType] : dict[AgentID, dict[str, Any]]]:
        super().reset(seed=seed, **kwargs)
        self._grid.reset()
        for agent in self._agents:
            agent.reset()
        observations = self._generate_observations()
        return observations, defaultdict(dict)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return spaces.Dict(
            {agent.index: agent.observation_space for agent in self._agents}
        )

    @property
    def action_space(self) -> gym.spaces.Space:
        return spaces.Dict({agent.index: agent.action_space for agent in self._agents})

    def _generate_observations(self) -> dict[AgentID, ObservationType]:
        pass
