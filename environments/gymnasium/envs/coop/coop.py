"""Maze environment for reinforcement learning."""

__credits__ = ["Eirik Reiestad"]
import logging
import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame as pg
from gymnasium import spaces

from environments import settings
from environments.gymnasium.utils import (
    Color,
    Direction,
    Position,
    State,
)
from .utils import TileType as TileType, Agents, Agent, AgentType

logging.basicConfig(level=logging.INFO)

AGENT_TILE_TYPE = {
    AgentType.AGENT0: TileType.AGENT0.value,
    AgentType.AGENT1: TileType.AGENT1.value,
}


class CoopEnv(gym.Env):
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
        self.max_steps = self.height * self.width

        self.screen_width = settings.SCREEN_WIDTH
        self.screen_height = settings.SCREEN_HEIGHT

        self._check_file_existence(
            "environments/gymnasium/data/coop/", settings.FILENAME
        )
        self._init_render_mode(render_mode)
        self._init_states("environments/gymnasium/data/coop/" + settings.FILENAME)
        self._init_spaces()

        self.rewards = {
            "goal": settings.GOAL_REWARD,
            "move": settings.MOVE_REWARD,
            "terminated": settings.TERMINATED_REWARD,
            "truncated": settings.TRUNCATED_REWARD,
        }

        self.steps = 0
        self.steps_beyond_terminated = None

        self.__post_init__()

    def __post_init__(self):
        """Post-initialization steps."""
        self.post_init = False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Takes a step in the environment.

        Parameters:
            action (int): The action to take.

        Returns:
            Tuple[np.ndarray, int, bool, bool, Dict[str, Any]]:
                - np.ndarray: The new state of the environment.
                - float: The reward for the action.
                - bool: Whether the episode is terminated.
                - bool: Whether the episode is truncated.
                - dict: Additional information.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self.steps += 1

        if self.steps >= self.max_steps:
            return (
                self.state.active_state,
                self.rewards["truncated"],
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
                self.update_state(new_full_state)
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
        if not self.post_init:
            self._init_render()
            self.post_init = True

        super().reset(seed=seed)
        render_mode = options.get("render_mode") if options else None
        self.render_mode = render_mode or self.render_mode

        self.steps = 0

        self._init_states()
        self._set_initial_positions(options)

        self.steps_beyond_terminated = None

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
        surf = pg.transform.scale(surf, (self.screen_height, self.screen_width))
        surf = pg.transform.flip(surf, True, False)
        surf = pg.transform.rotate(surf, 90)

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
        if hasattr(self, "screen") and self.screen:
            pg.display.quit()
            pg.quit()
            self.is_open = False

    def concat_state(self, states: list[np.ndarray]) -> tuple[np.ndarray, float, bool]:
        """Concatenates states from multiple agents into a single state."""
        agent0_state: np.ndarray = states[0]
        if len(states) == 1:
            agent1_state = agent0_state.copy()
        else:
            agent1_state: np.ndarray = states[1]

        agent0_position = np.where(agent0_state == AGENT_TILE_TYPE[AgentType.AGENT0])
        agent1_position = np.where(agent1_state == AGENT_TILE_TYPE[AgentType.AGENT1])
        obstacle_positions = np.where(agent0_state == TileType.OBSTACLE.value)

        agent0_position = int(agent0_position[0]), int(agent0_position[1])
        agent1_position = int(agent1_position[0]), int(agent1_position[1])

        state = np.zeros((self.height, self.width), dtype=np.float32)
        state[agent0_position[1], agent0_position[0]] = AGENT_TILE_TYPE[
            AgentType.AGENT0
        ]
        state[agent1_position[1], agent1_position[0]] = AGENT_TILE_TYPE[
            AgentType.AGENT1
        ]

        for x, y in zip(obstacle_positions[0], obstacle_positions[1]):
            state[y, x] = TileType.OBSTACLE.value

        if self._side_by_side(agent0_position, agent1_position):
            return state, self.rewards["goal"], True
        return state, 0.0, False

    def set_active_agent(self, agent: int):
        """Sets the active agent."""
        match agent:
            case 0:
                self.agents.active_agent = AgentType.AGENT0
            case 1:
                self.agents.active_agent = AgentType.AGENT1
            case _:
                raise ValueError(f"Invalid agent {agent}")

    def _side_by_side(
        self,
        pos0: tuple[int | float, int | float],
        pos1: tuple[int | float, int | float],
        radius: int = 1,
    ) -> bool:
        """Checks if the agents are side by side."""
        distance = abs(pos0[0] - pos1[0]) + abs(pos0[1] - pos1[1])
        return distance <= radius

    def _get_reward(self, collided: bool) -> float:
        """
        Calculates the reward for the current state.

        Parameters:
            collided (bool): Whether the agent collided.

        Returns:
            int: The reward.
        """
        if collided:
            return self.rewards["terminated"]
        else:
            return self.rewards["move"]

    def _move_agent(self, state: np.ndarray, action: int) -> Optional[np.ndarray]:
        """
        Moves the agent in the maze based on the action.

        Parameters:
            state (np.ndarray): The current state of the maze.
            action (int): The action to take.
            agent (str): The agent to move.

        Returns:
            Optional[np.ndarray]: The new state of the env, or None if the agent collided.
        """
        if self.agents.active_agent == AgentType.AGENT0:
            other_agent_position = self.agents.agent1.position
        else:
            other_agent_position = self.agents.agent0.position

        new_state = state.copy()
        new_agent_position = self.agents.active.position + Direction(action).to_tuple()

        if new_agent_position == other_agent_position:
            return None

        if self._is_within_bounds(new_agent_position) and self._is_not_obstacle(
            new_state, new_agent_position
        ):
            new_state[self.agents.active.position.x, self.agents.active.position.y] = (
                TileType.EMPTY.value
            )
            self.agents.active.position = new_agent_position
            new_state[self.agents.active.position.x, self.agents.active.position.y] = (
                AGENT_TILE_TYPE[self.agents.active_agent]
            )
            return new_state
        return None

    def _init_states(self, filename: str | None = None):
        """Initializes the maze states from the given file."""
        if not hasattr(self, "init_full_state") or self.init_full_state is None:
            if filename is None:
                raise ValueError("The maze file should be provided.")
            if not os.path.exists(filename):
                logging.info(f"Current working directory: {os.getcwd()}")
                raise FileNotFoundError(f"File {filename} does not exist.")

            with open(filename, "r") as f:
                env = [list(map(int, list(row.strip()))) for row in f.readlines()]
                self._validate_maze(env)

            self.init_full_state = np.array(env, dtype=np.uint8)

        full_state = np.array(self.init_full_state, dtype=np.uint8)
        partial_state = np.ndarray((7,), dtype=np.uint8)
        rgb_state = self._create_rgb_state()

        self.state = State(
            full=full_state,
            partial=partial_state,
            rgb=rgb_state,
            active=settings.STATE_TYPE,
        )

        self.init_agents = Agents(
            agent0=Agent(
                Position(
                    tuple(
                        np.argwhere(full_state == AGENT_TILE_TYPE[AgentType.AGENT0])[0]
                    )
                )
            ),
            agent1=Agent(
                Position(
                    tuple(
                        np.argwhere(full_state == AGENT_TILE_TYPE[AgentType.AGENT1])[0]
                    )
                )
            ),
            active_agent=AgentType.AGENT0,
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

    def _init_render_mode(self, render_mode: Optional[str]):
        """Initializes the render mode."""
        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render mode {render_mode}. "
                f"Available modes: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

    def _init_render(self):
        """Initializes rendering settings."""
        self.screen_width = settings.SCREEN_WIDTH
        self.screen_height = settings.SCREEN_HEIGHT

        pg.init()
        pg.display.init()
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.surface = pg.Surface((self.screen_width, self.screen_height))
        self.clock = pg.time.Clock()
        self.is_open = True

    def update_state(self, full_state: np.ndarray):
        """Updates the environment's state."""
        if full_state.shape != self.state.full.shape:
            raise ValueError(
                f"Invalid state shape. Expected {self.state.full.shape}, got {full_state.shape}"
            )
        self.state.full = full_state
        self.state.partial = self._create_partial_state()
        self.state.rgb = np.ndarray(
            (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT, 3), dtype=np.uint8
        )

    def _create_partial_state(self) -> np.ndarray:
        """Creates the partial state representation."""
        if self.agents.active_agent == AgentType.AGENT0:
            agent_position = self.agents.agent0.position
            goal_position = self.agents.agent1.position
        else:
            agent_position = self.agents.agent1.position
            goal_position = self.agents.agent0.position

        agent_distance = (goal_position - agent_position).to_tuple()
        goal_direction = goal_position - agent_position
        goal_direction = [goal_direction.x, goal_direction.y]

        distance = np.linalg.norm(agent_distance)
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
                *agent_position.to_tuple(),
                *goal_position.to_tuple(),
                int(distance_normalized),
                *direction_normalized,
            ],
            dtype=np.uint8,
        )

    def _check_file_existence(self, dir_path: str, filename: str):
        """Checks if the file exists."""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
        if not os.path.isfile(dir_path + filename):
            raise FileNotFoundError(f"File {filename} does not exist.")

    def _set_initial_positions(self, options: Optional[Dict[str, Any]]):
        """Sets the initial positions of the agent and goal."""
        if options and "agent0" in options:
            if "agent1" not in options:
                raise ValueError("The second agent's position is not set.")
            self.agents = Agents(
                agent0=Agent(Position(options["agent0"])),
                agent1=Agent(Position(options["agent1"])),
                active_agent=AgentType.AGENT0,
            )
        else:
            self.agents = Agents(
                agent0=Agent(self.init_agents.agent0.position),
                agent1=Agent(self.init_agents.agent1.position),
                active_agent=AgentType.AGENT0,
            )

    def _clear_start_goal_positions(self):
        """Clears the start and goal positions from the maze."""
        self.state.full[np.argwhere(self.state.full == TileType.AGENT0.value)] = (
            TileType.EMPTY.value
        )
        self.state.full[np.argwhere(self.state.full == TileType.AGENT1.value)] = (
            TileType.EMPTY.value
        )

    def _apply_color_masks(self, color_matrix: np.ndarray):
        """Applies color masks to the maze."""
        color_matrix[self.state.full == TileType.OBSTACLE.value] = Color.BLACK.value
        color_matrix[self.state.full == TileType.AGENT0.value] = Color.BLUE.value
        color_matrix[self.state.full == TileType.AGENT1.value] = Color.GREEN.value

    def _is_within_bounds(self, position: Position) -> bool:
        """Checks if the position is within maze bounds."""
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def _is_not_obstacle(self, state: np.ndarray, position: Position) -> bool:
        """Checks if the position is not an obstacle."""
        return state[int(position.x), int(position.y)] != TileType.OBSTACLE.value

    def _create_rgb_state(self) -> np.ndarray:
        """
        Creates an RGB representation of the maze state.

        Returns:
            np.ndarray: The RGB representation of the maze state.
        """
        return np.full(
            (settings.SCREEN_HEIGHT, settings.SCREEN_WIDTH, 3), Color.WHITE.value
        )

    def _validate_maze(self, maze: list[list[int]]):
        """Validates the maze file's content."""
        if not maze:
            raise ValueError("No data: Failed to read maze from file.")
        if len(maze) != self.height or len(maze[0]) != self.width:
            raise ValueError(
                f"Invalid maze size. Expected {self.height}x{self.width}, got {len(maze)}x{len(maze[0])}"
            )
        flatten_maze = np.array(maze).flatten()
        if np.all(flatten_maze != TileType.AGENT0.value):
            raise ValueError("The agent0 position is not set.")
        if np.all(flatten_maze != TileType.AGENT1.value):
            raise ValueError("The agent1 position is not set.")

    def get_active_state(self) -> np.ndarray:
        """Returns the active state of the environment."""
        return self.state.active_state

    @property
    def num_agents(self) -> int:
        """Returns the number of agents in the environment."""
        return 2
