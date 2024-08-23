"""Coop environment for reinforcement learning."""

__credits__ = ["Eirik Reiestad"]

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments import settings
from environments.gymnasium.utils import (
    Direction,
    FileHandler,
    Position,
    generate_random_position,
)

from .coop_renderer import CoopRenderer
from .coop_rewards import CoopRewards
from .coop_state import CoopState
from .env_utils import EnvUtils
from .utils import AGENT_TILE_TYPE, Agent, AgentType, DualAgents, TileType

logging.basicConfig(level=logging.INFO)


class CoopEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = "human"):
        self.height = settings.ENV_HEIGHT
        self.width = settings.ENV_WIDTH
        self.max_steps = self.height * self.width

        folder_name = "environments/gymnasium/data/coop/"
        FileHandler.file_exist(folder_name, settings.FILENAME)

        self.coop_renderer = CoopRenderer(
            settings.ENV_HEIGHT,
            settings.ENV_WIDTH,
            settings.SCREEN_WIDTH,
            settings.SCREEN_HEIGHT,
        )
        self.coop_renderer.init_render_mode(render_mode)

        filename = folder_name + settings.FILENAME
        self.state = CoopState(self.height, self.width, filename)
        self._init_spaces()
        self.coop_rewards = CoopRewards()

        self.steps = 0
        self.steps_beyond_terminated = None
        self._set_initial_positions(None)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self.steps += 1
        if self.steps >= self.max_steps:
            return (
                self.state.active_state,
                self.coop_rewards.truncated_reward,
                True,
                True,
                {"full_state": self.state.full},
            )

        collided = False
        terminated = False

        if not terminated:
            new_full_state = self._move_agent(self.state.full, action)
            collided = new_full_state is None
            if not collided and new_full_state is not None:
                self.state.update(
                    new_full_state,
                    self.agents.active.position,
                    self.agents.inactive.position,
                )
            terminated = collided or terminated
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
        else:
            if self.steps_beyond_terminated == 0:
                logging.warning(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

        reward = self.coop_rewards.get_individual_reward(collided)

        return (
            self.state.active_state,
            reward,
            terminated,
            False,
            {"full_state": self.state.full},
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        render_mode = options.get("render_mode") if options else None
        self.render_mode = render_mode or self.render_mode

        self.state.reset(self.agents.active_agent)
        self._set_initial_positions(options)

        self.steps = 0
        self.steps_beyond_terminated = None

        return self.state.active_state, {"state_type": settings.STATE_TYPE.value}

    def render(self, _render_mode: Optional[str] = None) -> Optional[np.ndarray]:
        self.coop_renderer.render(self.state.full, _render_mode)

    def close(self):
        self.coop_renderer.close()

    def set_active_agent(self, agent: AgentType):
        self.agents.active_agent = agent

    def concatenate_states(
        self, states: list[np.ndarray]
    ) -> tuple[np.ndarray, float, bool]:
        state, _ = self.state.concatenate_states(states)
        reward, terminated = self.coop_rewards.get_cooperative_reward(
            self.agents.active.position, self.agents.inactive.position
        )
        return state, reward, terminated

    def update_state(self, state: np.ndarray) -> np.ndarray:
        self.state.update(
            state, self.agents.active.position, self.agents.inactive.position
        )
        return self.state.active_state

    def get_active_state(self) -> np.ndarray:
        return self.state.active_state

    def _move_agent(self, state: np.ndarray, action: int) -> Optional[np.ndarray]:
        new_state = state.copy()
        new_agent_position = self.agents.active.position + Direction(action).tuple
        if EnvUtils.is_within_bounds(
            new_state, new_agent_position.x, new_agent_position.y
        ) and EnvUtils.is_not_obstacle(
            new_state, int(new_agent_position.x), int(new_agent_position.y)
        ):
            return self._move_agent_within_bounds(new_state, new_agent_position)
        return None

    def _move_agent_within_bounds(
        self, state: np.ndarray, agent_position: Position
    ) -> Optional[np.ndarray]:
        state[*self.agents.active.position.row_major_order] = TileType.EMPTY.value
        self.agents.active.position = agent_position
        agent_tile_type = AGENT_TILE_TYPE.get(self.agents.active_agent)
        if agent_tile_type is None:
            raise ValueError(f"Invalid agent type {self.agents.active_agent}")
        state[*self.agents.active.position.row_major_order] = agent_tile_type
        return state

    def _init_spaces(self):
        """Initializes the action and observation spaces."""
        if self.state.active_state is None:
            raise ValueError("The state should be set before initializing spaces.")

        self.action_space = spaces.Discrete(4)

        observation_shape = self.state.active_state.shape
        if settings.STATE_TYPE.value == "full":
            self.observation_space = spaces.Box(
                low=0, high=3, shape=observation_shape, dtype=np.uint8
            )
        elif settings.STATE_TYPE.value == "partial":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=self.state.partial.shape, dtype=np.uint8
            )
        elif settings.STATE_TYPE.value == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=self.state.rgb.shape, dtype=np.uint8
            )
        else:
            raise ValueError(f"Invalid state type {settings.STATE_TYPE.value}")

    def _set_initial_positions(self, options: Optional[Dict[str, Any]]):
        """Sets the initial positions of the agent and goal."""
        if options and "agent0" in options:
            agent0_position = Position(options["agent0"])
            agent1_position = Position(
                options.get(
                    "agent1",
                    generate_random_position(
                        self.width, self.height, [agent0_position]
                    ),
                )
            )
        else:
            agent0_position = self.state.get_agent_position(AgentType.AGENT0)
            agent1_position = (
                self.state.get_agent_position(AgentType.AGENT1)
                if options is None or "agent1" not in options
                else Position(options["agent1"])
            )
        agent0 = Agent(agent0_position)
        agent1 = Agent(agent1_position)
        self.agents = DualAgents(agent0, agent1, AgentType.AGENT0)

    @property
    def num_agents(self) -> int:
        return 2