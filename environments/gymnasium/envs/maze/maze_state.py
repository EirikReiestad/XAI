import logging
import os

import numpy as np

from environments import settings
from environments.gymnasium.utils import State, Color
from environments.gymnasium.envs.maze.utils import FullStateDataExtractor
from environments.gymnasium.utils import Position


class MazeState:
    def __init__(self, height: int, width: int, filename: str):
        self.height = height
        self.width = width
        self.init_full_state = None
        self.init_states(filename)

    def init_states(self, filename: str):
        self.init_full_state = self._load_env_from_file(filename)

        full_state = self.init_full_state
        partial_state = np.ndarray((7,), dtype=np.uint8)
        # partial_state = self._create_partial_state()
        rgb_state = self._create_rgb_state()

        self.state = State(
            full=full_state,
            partial=partial_state,
            rgb=rgb_state,
            active=settings.STATE_TYPE,
        )

    def update(self, new_full_state: np.ndarray):
        self.state.full = new_full_state
        self.state.partial = self._create_partial_state(full_state=new_full_state)
        self.state.rgb = self._create_rgb_state()

    def _create_partial_state(self, full_state: np.ndarray) -> np.ndarray:
        agent_position = FullStateDataExtractor.get_agent_position(full_state)
        goal_position = FullStateDataExtractor.get_goal_position(full_state)

        goal_distance = goal_position - agent_position
        goal_direction = [goal_distance.x, goal_distance.y]

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

    def _create_rgb_state(self) -> np.ndarray:
        return np.full(
            (settings.SCREEN_HEIGHT, settings.SCREEN_WIDTH, 3), Color.WHITE.value
        )

    def _load_env_from_file(self, filename: str) -> np.ndarray:
        if not os.path.exists(filename):
            logging.info(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"File {filename} does not exist.")

        with open(filename, "r") as f:
            maze = [list(map(int, list(row.strip()))) for row in f.readlines()]

        return np.array(maze, dtype=np.uint8)

    def get_agent_position(self) -> Position:
        return FullStateDataExtractor.get_agent_position(self.state.full)

    @property
    def active_state(self) -> np.ndarray:
        return self.state.active_state

    @property
    def full(self) -> np.ndarray:
        return self.state.full

    @property
    def partial(self) -> np.ndarray:
        return self.state.partial

    @property
    def rgb(self) -> np.ndarray:
        return self.state.rgb

    @property
    def initial_agent_position(self) -> Position:
        if self.init_full_state is None:
            raise ValueError("Initial full state is not set yet.")
        return FullStateDataExtractor.get_agent_position(self.init_full_state)

    @property
    def initial_goal_position(self) -> Position:
        if self.init_full_state is None:
            raise ValueError("Initial full state is not set yet.")
        return FullStateDataExtractor.get_goal_position(self.init_full_state)
