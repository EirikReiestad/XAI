"""
Maze system
"""

__credits__ = ["Eirik Reiestad"]

import os
import logging
from typing import Optional, Tuple

import numpy as np
import pygame as pg

import gymnasium as gym
from gymnasium import spaces

from .utils import MazeTileType as TileType
from environments.gymnasium.utils import (
    Color,
    Position,
    generate_random_position,
    Direction,
)
from environments import settings

from environments.gymnasium.utils import State

logging.basicConfig(level=logging.INFO)


class MazeEnv(gym.Env):
    """
    ## Description
    This class implements a simple maze environment. The goal is to reach the goal position

    ## Action Space

    The action is a ´ndarray´ with ´(1,) which can take values ´{0, 1, 2, 3}´ indicating the direction to move.
    ## Observation Space

    The observation is a ´ndarray´ with shape ´(height, width)´ with the current state of the maze.

    ## Rewards

    The reward is ´1´ if the agent reaches the goal position, otherwise ´0´.

    ## Starting State

    The starting state is a random position in the maze.
    The goal is a random position in the maze.

    ## Episode Termination

    The episode ends if any one of the following occurs:
    1. The agent reaches the goal position.
    2. The agent reaches the maximum number of steps.

    ## Arguments

    Optional arguments
    - max_steps: The maximum number of steps.
    - render_mode: The render mode.
    - start: The start position.
    - goal: The goal position.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    action_space: spaces.Discrete
    observation_space: spaces.Box

    state: State

    def __init__(self, render_mode: Optional[str] = "human"):
        self.height = settings.MAZE_HEIGHT
        self.width = settings.MAZE_WIDTH
        self.max_steps = (self.height * self.width) * 2

        dir_path = "environments/gymnasium/data/maze/"
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
        filename = settings.FILENAME
        filename = dir_path + filename

        self._init_states(filename)
        self._init_render(render_mode)
        self._init_spaces()

        self.rewards = {
            "goal": settings.GOAL_REWARD,
            "move": settings.MOVE_REWARD,
            "terminated": settings.TERMINATED_REWARD,
            "truncated": settings.TRUNCATED_REWARD,
        }

        self.steps = 0

        self.steps_beyond_terminated = None

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, dict]:
        """
        This method is called to take a step in the environment.
        Parameters:
            action: The action to take.

        Returns:
            np.ndarray: The new state of the environment.
            int: The reward of the action.
            bool: A boolean indicating if the episode is terminated.
            bool: A boolean indicating if the episode is truncated.
            dict: Additional information.
        """
        if self.action_space.contains(action) is False:
            raise ValueError(f"Invalid action {action}")

        self.steps += 1

        if self.steps >= self.max_steps:
            return self.state.active_state, self.rewards["truncated"], True, True, {}

        terminated = self.agent == self.goal
        collided = False

        if not terminated:
            new_state = self._move_agent(self.state.full, action)
            collided = new_state is None

            terminated = collided or terminated

            if not collided and new_state is not None:
                if self.state.active_state == StateType.RGB.value:
                    new_state = self.render()
                self.state.update_active_state(new_state)
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

        reward = self._get_reward(collided)
        return self.state.active_state, reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to the initial state.

        Parameters:
            seed: The seed to use.
            options: Additional options.

        Returns:
            np.ndarray: The initial state of the environment.
        """

        render_mode = options.get("render_mode") if options is not None else None

        self.render_mode = (
            render_mode
            if render_mode is not None and render_mode in self.metadata["render_modes"]
            else self.render_mode
        )

        if options is not None and "start" in options:
            self.agent = Position(options["start"])
            if "goal" in options:
                self.goal = Position(options["goal"])
            else:
                if self.agent is None:
                    raise ValueError("The agent position is not set.")

                self.goal = generate_random_position(
                    self.width, self.height, [self.agent]
                )
        else:
            if options is not None and "goal" in options:
                self.goal = Position(options["goal"])
                self.agent = self.init_agent
            else:
                self.agent = self.init_agent
                self.goal = self.init_goal

        self.steps = 0

        self.state.full[self.agent.y, self.agent.x] = TileType.EMPTY.value
        self.state.full[self.goal.y, self.goal.x] = TileType.EMPTY.value

        for agent in np.argwhere(self.state.full == TileType.START.value):
            self.state.full[agent[0], agent[1]] = TileType.EMPTY.value
        for goal in np.argwhere(self.state.full == TileType.END.value):
            self.state.full[goal[0], goal[1]] = TileType.EMPTY.value

        self.state.full[self.init_agent.y, self.init_agent.x] = TileType.START.value
        self.state.full[self.init_goal.y, self.init_goal.x] = TileType.END.value
        self.steps_beyond_terminated = None

        if np.where(self.state.full == TileType.END.value)[0].size == 0:
            raise ValueError("The goal position is not set.")

        return self.state.active_state, {"state_type": settings.STATE_TYPE.value}

    def render(self, _render_mode: Optional[str] = None) -> np.ndarray | None:
        render_mode = _render_mode if _render_mode is not None else self.render_mode

        color_matrix = np.full((self.height, self.width, 3), Color.WHITE.value)

        obstacle_mask = self.state.full == TileType.OBSTACLE.value
        agent_mask = self.state.full == TileType.START.value
        goal_mask = self.state.full == TileType.END.value

        color_matrix[obstacle_mask] = Color.BLACK.value
        color_matrix[agent_mask] = Color.BLUE.value
        color_matrix[goal_mask] = Color.GREEN.value

        surf = pg.surfarray.make_surface(color_matrix)
        surf = pg.transform.scale(surf, (self.screen_width, self.screen_height))
        surf = pg.transform.flip(surf, True, False)

        if render_mode == "rgb_array":
            self.surface.blit(surf, (0, 0))
            return pg.surfarray.array3d(self.surface)
        elif render_mode == "human":
            self.screen.blit(surf, (0, 0))
            pg.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pg.display.flip()
        return None

    def close(self):
        if self.screen is not None:
            pg.display.quit()
            pg.quit()
            self.is_open = False

    def _get_reward(self, collided: bool = False) -> int:
        """Calculate the reward of the current state.

        Parameters:
            colided: bool
                If the agent collided or not

        Returns:
            int: a reward
        """
        if self.agent == self.goal:
            return self.rewards["goal"]
        elif collided:
            return self.rewards["terminated"]
        else:
            return self.rewards["move"]

    def _move_agent(self, state: np.ndarray, action: int) -> np.ndarray | None:
        """
        Move the agent in the maze.
        Parameters:
            state: The current state of the maze.
            action: The action to take.

        Returns
            The new state of the maze. Return None if the state are the same, meaning the agent collided.
        """

        new_state: np.ndarray = state.copy()
        new_agent: Position = self.agent + Direction(action).to_tuple()
        if (
            new_agent.x >= 0
            and new_agent.x < self.width
            and new_agent.y >= 0
            and new_agent.y < self.height
        ):
            if new_state[int(new_agent.y), int(new_agent.y)] == TileType.OBSTACLE.value:
                return new_state
            new_state[self.agent.y, self.agent.x] = TileType.EMPTY.value
            self.agent = new_agent
            new_state[self.agent.y, self.agent.x] = TileType.START.value
            return new_state
        else:
            return None

    def _init_states(self, filename):
        if not os.path.exists(filename):
            logging.info(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"File {filename} does not exist.")

        with open(filename, "r") as f:
            maze = f.readlines()
            maze = [list(map(int, list(row.strip()))) for row in maze]

        full_state = np.array(maze, dtype=np.uint8)
        partial_state = full_state.flatten()
        rgb_state = np.ndarray((self.height, self.width, 3), dtype=np.uint8)

        self.state = State(
            full=full_state,
            partial=partial_state,
            rgb=rgb_state,
            active=settings.STATE_TYPE,
        )

        self.init_agent = Position(
            tuple(map(int, np.argwhere(self.state.full == TileType.START.value)[0]))
        )
        self.init_goal = Position(
            tuple(map(int, np.argwhere(self.state.full == TileType.END.value)[0]))
        )

        if maze in [None, [], ""]:
            logging.error(f"No data: Failed to read maze from file {filename}.")
            raise ValueError("No data: Failed to read maze from file.")

        if len(maze) != self.height or len(maze[0]) != self.width:
            logging.error(
                f"Invalid maze size. Expected {self.height}x{self.width}, got {len(maze)}x{len(maze[0])}"
            )
            raise ValueError("Invalid maze size.")

        flatten_maze = np.array(maze).flatten()

        if np.where(flatten_maze == TileType.END.value)[0].size == 0:
            logging.error("The goal position is not set.")
            raise ValueError("The goal position is not set.")
        if np.where(flatten_maze == TileType.START.value)[0].size == 0:
            logging.error("The start position is not set.")
            raise ValueError("The start position is not set.")

    def _init_spaces(self):
        if self.state.active_state is None:
            raise ValueError("The state should be set before the spaces.")

        self.action_space = spaces.Discrete(4)

        match settings.STATE_TYPE.value:
            case "full":
                self.observation_space: spaces.Box = gym.spaces.Box(
                    low=0, high=3, shape=self.state.full.shape, dtype=np.uint8
                )
            case "partial":
                self.observation_space: spaces.Box = gym.spaces.Box(
                    low=0, high=3, shape=self.state.partial.shape, dtype=np.uint8
                )
            case "rgb":
                self.observation_space: spaces.Box = gym.spaces.Box(
                    low=0, high=255, shape=self.state.rgb.shape, dtype=np.uint8
                )
            case _:
                raise ValueError(f"Invalid state type {settings.STATE_TYPE}")

    def _init_render(self, render_mode: str | None):
        if render_mode is None:
            render_mode = "human"
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode {render_mode}")

        self.render_mode = render_mode

        self.screen_width = settings.SCREEN_WIDTH
        self.screen_height = settings.SCREEN_HEIGHT

        pg.init()
        pg.display.init()
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.surface = pg.Surface((self.screen_width, self.screen_height))
        self.clock = pg.time.Clock()
        self.is_open = True
