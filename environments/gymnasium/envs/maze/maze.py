"""
Maze system
"""

__credits__ = ["Eirik Reiestad"]

import os
import logging
from typing import Optional, Tuple, Union

import numpy as np
import pygame as pg

import gymnasium as gym
from gymnasium import spaces
import gymnasium.logger as logger

from .utils import MazeTileType as TileType
from environments.gymnasium.utils import (
    Color,
    Position,
    generate_random_position, Direction)

logging.basicConfig(level=logging.INFO)


class MazeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
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
    - height: The height of the maze.
    - width: The width of the maze.
    - max_steps: The maximum number of steps.
    - render_mode: The render mode.
    - start: The start position.
    - goal: The goal position.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None):
        self.height = 5
        self.width = 5
        self.max_steps = 100

        self.agent = None
        self.goal = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(self.height, self.width), dtype=np.uint8
        )

        self.rewards = {
            "goal": 1,
            "truncate": -1,
        }

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.surface = None
        self.clock = None
        self.is_open = True

        dir_path = "environments/gymnasium/data/maze/"

        if not os.path.exists(dir_path):
            raise FileNotFoundError(
                f"Directory {dir_path} does not exist.")

        filename = "maze-0-10-10.txt"
        filename = dir_path + filename

        ok = self._load_init_state(filename)
        if not ok:
            raise ValueError("Failed to load initial state.")

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
        assert self.action_space.contains(
            action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        self.steps += 1

        if self.steps >= self.max_steps:
            return self.state, self.rewards["truncate"], True, True, {}

        terminated = self.agent == self.goal
        reward = self.rewards["goal"] if terminated else 0

        if not terminated:
            self.state = self._move_agent(self.state, action)
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

        return self.state, reward, terminated, False, {}

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None
              ) -> np.ndarray:
        """
        Reset the environment to the initial state.

        Parameters:
            seed: The seed to use.
            options: Additional options.

        Returns:
            np.ndarray: The initial state of the environment.
        """

        if options is not None and "start" in options:
            self.agent = Position(options["start"])
            if "goal" in options:
                if options["goal"] == self.agent:
                    raise ValueError(
                        "The goal position is the same as the agent position.")
                self.goal = Position(options["goal"])
            else:
                generate_random_position(
                    self.width, self.height, self.agent)
        else:
            if options is not None and "goal" in options:
                self.goal = Position(options["goal"])
                self.agent = self.init_agent
            else:
                self.agent = self.init_agent
                self.goal = self.init_goal

        self.steps = 0

        self.state[self.agent.y, self.agent.x] = TileType.EMPTY.value
        self.state[self.goal.y, self.goal.x] = TileType.EMPTY.value

        for agent in np.argwhere(self.state == TileType.START.value):
            self.state[agent[0], agent[1]] = TileType.EMPTY.value
        for goal in np.argwhere(self.state == TileType.END.value):
            self.state[goal[0], goal[1]] = TileType.EMPTY.value

        self.state[self.init_agent.y, self.init_agent.x] = TileType.START.value
        self.state[self.init_goal.y, self.init_goal.x] = TileType.END.value
        self.steps_beyond_terminated = None

        if np.where(self.state == TileType.END.value)[0].size == 0:
            raise ValueError("The goal position is not set.")

        return self.state, {}

    def render(self, render_mode: Optional[str] = None):
        render_mode = render_mode if render_mode is not None else self.render_mode
        if render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "No render mode was set. Using 'human' as default.")
            render_mode = "human"

        if self.screen is None or self.surface is None:
            pg.init()
            pg.display.init()
            self.screen = pg.display.set_mode(
                (self.screen_width, self.screen_height))
            self.surface = pg.Surface(
                (self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pg.time.Clock()

        if self.state is None:
            return None

        color_matrix = np.full(
            (self.height, self.width, 3), Color.WHITE.value)

        obstacle_mask = self.state == TileType.OBSTACLE.value
        agent_mask = self.state == TileType.START.value
        goal_mask = self.state == TileType.END.value

        color_matrix[obstacle_mask] = Color.BLACK.value
        color_matrix[agent_mask] = Color.BLUE.value
        color_matrix[goal_mask] = Color.GREEN.value

        surf = pg.surfarray.make_surface(color_matrix)
        surf = pg.transform.scale(
            surf, (self.screen_width, self.screen_height))
        surf = pg.transform.flip(surf, True, False)

        if render_mode == "rgb_array":
            self.surface.blit(surf, (0, 0))
            return np.transpose(
                np.array(pg.surfarray.pixels3d(self.surface)), axes=(1, 0, 2))
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

    def _move_agent(self, state: (np.ndarray, Position), action: int) -> list[np.ndarray, Position]:
        """
        Move the agent in the maze.
        :Arguments
        - state: The current state of the maze.
        - action: The action to take.
        :Returns
        - The new state of the maze.
        - A boolean indicating if the agent reached the goal.
        """

        new_state = state.copy()
        new_agent = self.agent + Direction(action).to_tuple()
        if new_agent.x >= 0 and new_agent.x < self.width and new_agent.y >= 0 and new_agent.y < self.height:
            if new_state[new_agent.y, new_agent.x] == TileType.OBSTACLE.value:
                return new_state
            new_state[self.agent.y, self.agent.x] = TileType.EMPTY.value
            self.agent = new_agent
            new_state[self.agent.y, self.agent.x] = TileType.START.value
        return new_state

    def _load_init_state(self, filename) -> bool:
        if not os.path.exists(filename):
            logging.info(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"File {filename} does not exist.")

        with open(filename, "r") as f:
            maze = f.readlines()
            maze = [list(map(int, list(row.strip()))) for row in maze]

            self.state = np.array(maze, dtype=np.uint8)

            self.init_agent = Position(tuple(
                map(int, np.argwhere(self.state == TileType.START.value)[0])))
            self.init_goal = Position(tuple(
                map(int, np.argwhere(self.state == TileType.END.value)[0])))

            if maze in [None, [], ""]:
                logging.error(
                    f"No data: Failed to read maze from file {filename}.")
                return False

            if len(maze) != self.height or len(maze[0]) != self.width:
                logging.error(
                    f"Invalid maze size. Expected {self.height}x{self.width}, got {len(maze)}x{len(maze[0])}")
                return False

            flatten_maze = np.array(maze).flatten()

            if np.where(flatten_maze == TileType.END.value)[0].size == 0:
                logging.error("The goal position is not set.")
                return False
            if np.where(flatten_maze == TileType.START.value)[0].size == 0:
                logging.error("The start position is not set.")
                return False

            return True
        return False
