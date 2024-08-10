"""Maze environment for reinforcement learning."""

__credits__ = ["Eirik Reiestad"]

import os
import logging
from typing import Optional, Tuple, Dict, Any

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
    A maze environment where the agent must reach a goal position.

    Attributes:
        metadata (dict): Metadata for rendering.
        action_space (spaces.Discrete): The action space of the environment.
        observation_space (spaces.Box): The observation space of the environment.
        state (State): The current state of the maze.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = "human"):
        """
        Initializes the maze environment.

        Parameters:
            render_mode (Optional[str]): The render mode for the environment.
        """
        self.height = settings.MAZE_HEIGHT
        self.width = settings.MAZE_WIDTH
        self.max_steps = (self.height * self.width) * 2

        self._check_file_existence(
            "environments/gymnasium/data/maze/", settings.FILENAME
        )
        self._init_render(render_mode if render_mode else "human")
        self._init_states("environments/gymnasium/data/maze/" + settings.FILENAME)
        self._init_spaces()

        self.rewards = {
            "goal": settings.GOAL_REWARD,
            "move": settings.MOVE_REWARD,
            "terminated": settings.TERMINATED_REWARD,
            "truncated": settings.TRUNCATED_REWARD,
        }

        self.steps = 0
        self.steps_beyond_terminated = None

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, Dict[str, Any]]:
        """
        Takes a step in the environment.

        Parameters:
            action (int): The action to take.

        Returns:
            Tuple[np.ndarray, int, bool, bool, Dict[str, Any]]:
                - np.ndarray: The new state of the environment.
                - int: The reward for the action.
                - bool: Whether the episode is terminated.
                - bool: Whether the episode is truncated.
                - dict: Additional information.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self.steps += 1

        if self.steps >= self.max_steps:
            return self.state.active_state, self.rewards["truncated"], True, True, {}

        terminated = self.agent == self.goal
        collided = False

        if not terminated:
            new_full_state = self._move_agent(self.state.full, action)
            collided = new_full_state is None
            if not collided and new_full_state is not None:
                self._update_state(new_full_state)
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

        reward = self._get_reward(collided)
        return self.state.active_state, reward, terminated, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to the initial state.

        Parameters:
            seed (Optional[int]): The seed for random number generation.
            options (Optional[Dict[str, Any]]): Options for resetting the environment.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - np.ndarray: The initial state of the environment.
                - dict: Additional information about the state.
        """
        render_mode = options.get("render_mode") if options else None
        self.render_mode = render_mode or self.render_mode

        self._set_initial_positions(options)
        self.steps = 0

        self.state.full[self.agent.y, self.agent.x] = TileType.EMPTY.value
        self.state.full[self.goal.y, self.goal.x] = TileType.EMPTY.value

        self._clear_start_goal_positions()

        self.state.full[self.init_agent.y, self.init_agent.x] = TileType.START.value
        self.state.full[self.init_goal.y, self.init_goal.x] = TileType.END.value
        self.steps_beyond_terminated = None

        if np.where(self.state.full == TileType.END.value)[0].size == 0:
            raise ValueError("The goal position is not set.")

        return self.state.active_state, {"state_type": settings.STATE_TYPE.value}

    def render(self, _render_mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Renders the current state of the environment.

        Parameters:
            _render_mode (Optional[str]): The render mode.

        Returns:
            Optional[np.ndarray]: The rendered image or None.
        """
        render_mode = _render_mode or self.render_mode

        color_matrix = np.full((self.height, self.width, 3), Color.WHITE.value)
        self._apply_color_masks(color_matrix)

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
        """Closes the environment and the rendering window."""
        if self.screen:
            pg.display.quit()
            pg.quit()
            self.is_open = False

    def _get_reward(self, collided: bool) -> int:
        """
        Calculates the reward for the current state.

        Parameters:
            collided (bool): Whether the agent collided.

        Returns:
            int: The reward.
        """
        if self.agent == self.goal:
            return self.rewards["goal"]
        elif collided:
            return self.rewards["terminated"]
        else:
            return self.rewards["move"]

    def _move_agent(self, state: np.ndarray, action: int) -> Optional[np.ndarray]:
        """
        Moves the agent in the maze based on the action.

        Parameters:
            state (np.ndarray): The current state of the maze.
            action (int): The action to take.

        Returns:
            Optional[np.ndarray]: The new state of the maze, or None if the agent collided.
        """
        new_state = state.copy()
        new_agent = self.agent + Direction(action).to_tuple()
        if self._is_within_bounds(new_agent) and self._is_not_obstacle(
            new_state, new_agent
        ):
            new_state[self.agent.y, self.agent.x] = TileType.EMPTY.value
            self.agent = new_agent
            new_state[self.agent.y, self.agent.x] = TileType.START.value
            return new_state
        return None

    def _init_states(self, filename: str):
        """Initializes the maze states from the given file."""
        if not os.path.exists(filename):
            logging.info(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"File {filename} does not exist.")

        with open(filename, "r") as f:
            maze = [list(map(int, list(row.strip()))) for row in f.readlines()]
            self._validate_maze(maze)

        full_state = np.array(maze, dtype=np.uint8)
        partial_state = np.ndarray((7,), dtype=np.uint8)
        rgb_state = self._create_rgb_state()

        self.state = State(
            full=full_state,
            partial=partial_state,
            rgb=rgb_state,
            active=settings.STATE_TYPE,
        )

        self.init_agent = Position(
            tuple(np.argwhere(self.state.full == TileType.START.value)[0])
        )
        self.init_goal = Position(
            tuple(np.argwhere(self.state.full == TileType.END.value)[0])
        )

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

    def _init_render(self, render_mode: str):
        """Initializes rendering settings."""
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

    def _update_state(self, new_full_state: np.ndarray):
        """Updates the environment's state."""
        self.state.full = new_full_state
        self.state.partial = self._create_partial_state()
        self.state.rgb = self.render("rgb_array") or self.state.rgb

    def _create_partial_state(self) -> np.ndarray:
        """Creates the partial state representation."""
        agent_position = [self.agent.y, self.agent.x]
        goal_position = [self.goal.y, self.goal.x]

        goal_distance = self.goal - self.agent
        goal_direction = [goal_distance.y, goal_distance.x]

        distance = np.linalg.norm(goal_direction)
        distance_normalized = np.clip(
            distance / np.sqrt(self.height**2 + self.width**2) * 255, 0, 255
        )

        direction_offset = 128
        direction_normalized = np.clip(
            [
                (val / np.sqrt(self.height**2 + self.width**2)) * 127 + direction_offset
                for val in goal_direction
            ],
            0,
            255,
        )

        return np.array(
            [
                *agent_position,
                *goal_position,
                int(distance_normalized),
                *map(int, direction_normalized),
            ],
            dtype=np.uint8,
        )

    def _check_file_existence(self, dir_path: str, filename: str):
        """Checks if the file exists."""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")

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
            self.agent = self.init_agent
            self.goal = (
                self.init_goal
                if options is None or "goal" not in options
                else Position(options["goal"])
            )

    def _clear_start_goal_positions(self):
        """Clears the start and goal positions from the maze."""
        self.state.full[np.argwhere(self.state.full == TileType.START.value)] = (
            TileType.EMPTY.value
        )
        self.state.full[np.argwhere(self.state.full == TileType.END.value)] = (
            TileType.EMPTY.value
        )

    def _apply_color_masks(self, color_matrix: np.ndarray):
        """Applies color masks to the maze."""
        color_matrix[self.state.full == TileType.OBSTACLE.value] = Color.BLACK.value
        color_matrix[self.state.full == TileType.START.value] = Color.BLUE.value
        color_matrix[self.state.full == TileType.END.value] = Color.GREEN.value

    def _is_within_bounds(self, position: Position) -> bool:
        """Checks if the position is within maze bounds."""
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def _is_not_obstacle(self, state: np.ndarray, position: Position) -> bool:
        """Checks if the position is not an obstacle."""
        return state[int(position.y), int(position.x)] != TileType.OBSTACLE.value

    def _create_rgb_state(self) -> np.ndarray:
        """
        Creates an RGB representation of the maze state.

        Returns:
            np.ndarray: The RGB representation of the maze state.
        """
        color_matrix = np.full((self.height, self.width, 3), Color.WHITE.value)
        surf = pg.surfarray.make_surface(color_matrix)
        surf = pg.transform.scale(surf, (self.screen_width, self.screen_height))
        surf = pg.transform.flip(surf, True, False)
        self.surface.blit(surf, (0, 0))
        rgb_state = pg.surfarray.array3d(self.surface)
        return rgb_state

    def _validate_maze(self, maze: list[list[int]]):
        """Validates the maze file's content."""
        if not maze:
            raise ValueError("No data: Failed to read maze from file.")
        if len(maze) != self.height or len(maze[0]) != self.width:
            raise ValueError(
                f"Invalid maze size. Expected {self.height}x{self.width}, got {len(maze)}x{len(maze[0])}"
            )
        flatten_maze = np.array(maze).flatten()
        if np.all(flatten_maze != TileType.END.value):
            raise ValueError("The goal position is not set.")
        if np.all(flatten_maze != TileType.START.value):
            raise ValueError("The start position is not set.")
