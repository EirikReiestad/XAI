"""Maze environment for reinforcement learning."""

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

from .maze_renderer import MazeRenderer
from .maze_rewards import MazeRewards
from .maze_state import MazeState
from .maze_utils import MazeUtils
from .utils import TileType

logging.basicConfig(level=logging.INFO)


class MazeEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = "human"):
        self.height = settings.ENV_HEIGHT
        self.width = settings.ENV_WIDTH
        self.max_steps = self.height * self.width

        FileHandler.file_exist("environments/gymnasium/data/maze/", settings.FILENAME)

        self.maze_renderer = MazeRenderer(
            settings.ENV_HEIGHT,
            settings.ENV_WIDTH,
            settings.SCREEN_WIDTH,
            settings.SCREEN_HEIGHT,
        )
        self.maze_renderer.init_render_mode(render_mode)

        filename = "environments/gymnasium/data/maze/" + settings.FILENAME
        self.state = MazeState(self.height, self.width, filename)
        self._init_spaces()
        self.maze_rewards = MazeRewards()

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
                self.maze_rewards.truncated_reward,
                True,
                True,
                {},
            )

        terminated = self.agent == self.goal
        collided = False

        if not terminated:
            new_full_state = self._move_agent(self.state.full, action)
            collided = new_full_state is None
            if not collided and new_full_state is not None:
                self.state.update(new_full_state)
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

        reward = self.maze_rewards.get_reward(self.agent, self.goal, collided)
        return self.state.active_state, reward, terminated, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        render_mode = options.get("render_mode") if options else None
        self.render_mode = render_mode or self.render_mode

        self._set_initial_positions(options)

        self.steps = 0
        self.steps_beyond_terminated = None

        return self.state.active_state, {"state_type": settings.STATE_TYPE.value}

    def render(self, _render_mode: Optional[str] = None) -> Optional[np.ndarray]:
        self.maze_renderer.render(self.state.full, _render_mode)

    def close(self):
        self.maze_renderer.close()

    def _move_agent(self, state: np.ndarray, action: int) -> Optional[np.ndarray]:
        new_state = state.copy()
        new_agent = self.agent + Direction(action).to_tuple()
        if MazeUtils.is_within_bounds(
            new_state, new_agent.x, new_agent.y
        ) and MazeUtils.is_not_obstacle(new_state, int(new_agent.x), int(new_agent.y)):
            return self._move_agent_within_bounds(new_state, new_agent)
        return None

    def _move_agent_within_bounds(
        self, state: np.ndarray, agent
    ) -> Optional[np.ndarray]:
        state[self.agent.x, self.agent.y] = TileType.EMPTY.value
        self.agent = agent
        state[self.agent.x, self.agent.y] = TileType.START.value
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
        if options and "start" in options:
            self.agent = Position(options["start"])
            self.goal = Position(
                options.get(
                    "goal",
                    generate_random_position(self.width, self.height, [self.agent]),
                )
            )
        else:
            self.agent = self.state.initial_agent_position
            self.goal = (
                self.state.initial_goal_position
                if options is None or "goal" not in options
                else Position(options["goal"])
            )
