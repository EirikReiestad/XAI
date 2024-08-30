import logging
import os

import numpy as np

from environments import settings
from environments.gymnasium.envs.maze.utils import (
    FullStateDataExtractor,
    FullStateDataModifier,
)
from environments.gymnasium.utils import Position, State, StateType
from utils import Color


class MazeState:
    def __init__(self, height: int, width: int, filename: str):
        self.height = height
        self.width = width
        self.init_states(filename)

    def init_states(self, filename: str):
        self.init_full_state = self._load_env_from_file(filename)

        full_state = self.init_full_state
        partial_state = self._create_empty_partial_state()
        # partial_state = self._create_partial_state()
        rgb_state = self._create_rgb_state()

        self.state = State(
            full=full_state,
            partial=partial_state,
            rgb=rgb_state,
            active=settings.STATE_TYPE,
        )

    def reset(self):
        self.state.full = self.init_full_state
        agent_position = FullStateDataExtractor.get_agent_position(self.init_full_state)
        goal_position = FullStateDataExtractor.get_goal_position(self.init_full_state)
        obstacle_positions = FullStateDataExtractor.get_obstacle_positions(
            self.init_full_state
        )
        self.state.partial = self._create_partial_state(
            agent_position=agent_position,
            goal_position=goal_position,
            obstacle_positions=obstacle_positions,
        )
        self.state.rgb = self._create_rgb_state()

    def update(
        self,
        new_full_state: np.ndarray,
        agent_position: Position,
        goal_position: Position,
    ):
        self.state.full = new_full_state
        obstacle_positions = FullStateDataExtractor.get_obstacle_positions(
            self.state.full
        )
        self.state.partial = self._create_partial_state(
            agent_position=agent_position,
            goal_position=goal_position,
            obstacle_positions=obstacle_positions,
        )
        self.state.rgb = self._create_rgb_state()

    def _create_partial_state(
        self,
        agent_position: Position,
        goal_position: Position,
        obstacle_positions: list[Position],
    ) -> np.ndarray:
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

        obstacle_int_positions = []
        for obstacle_position in obstacle_positions:
            obstacle_int_positions.extend([*obstacle_position])

        return np.array(
            [
                *agent_position,
                *goal_position,
                int(distance_normalized),
                *map(int, direction_normalized),
                *obstacle_int_positions,
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

    def get_all_possible_states(self) -> np.ndarray:
        if settings.STATE_TYPE == StateType.FULL:
            return self._get_all_possible_full_states()
        elif settings.STATE_TYPE == StateType.PARTIAL:
            return self._get_all_possible_partial_states()
        elif settings.STATE_TYPE == StateType.RGB:
            raise NotImplementedError("RGB state type not implemented yet.")
        raise ValueError(f"Unknown state type: {settings.STATE_TYPE}")

    def _get_all_possible_full_states(self) -> np.ndarray:
        clean_agent_state = self.init_full_state.copy()
        clean_agent_state = FullStateDataModifier.remove_agent(clean_agent_state)
        states = np.ndarray(
            (self.height, self.width, *self.full_state_size), dtype=np.uint8
        )
        for y in range(self.height):
            for x in range(self.width):
                state = clean_agent_state.copy()
                agent_position = Position(x, y)
                if FullStateDataExtractor.is_empty_tile(
                    clean_agent_state, agent_position
                ):
                    state = FullStateDataModifier.place_agent(state, agent_position)
                else:
                    state = self._create_empty_full_state()
                states[agent_position.row_major_order] = state
        return states

    def _get_all_possible_partial_states(self) -> np.ndarray:
        clean_agent_state = self.init_full_state.copy()
        clean_agent_state = FullStateDataModifier.remove_agent(clean_agent_state)
        goal_position = FullStateDataExtractor.get_goal_position(self.init_full_state)
        obstacle_positions = FullStateDataExtractor.get_obstacle_positions(
            self.init_full_state
        )
        states = np.ndarray(
            (self.height, self.width, *self.partial_state_size), dtype=np.uint8
        )
        for y in range(self.height):
            for x in range(self.width):
                agent_position = Position(x, y)
                if FullStateDataExtractor.is_empty_tile(
                    clean_agent_state, agent_position
                ):
                    state = self.state.partial = self._create_partial_state(
                        agent_position=agent_position,
                        goal_position=goal_position,
                        obstacle_positions=obstacle_positions,
                    )
                else:
                    state = self._create_empty_partial_state()
                states[agent_position.row_major_order] = state
        return states

    def _create_empty_partial_state(self):
        return np.ndarray(self.partial_state_size, dtype=np.uint8)

    def _create_empty_full_state(self):
        return np.zeros((self.height, self.width), dtype=np.uint8)

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

    @property
    def partial_state_size(self) -> np.ndarray:
        agent_position = 2
        goal_position = 2
        distance = 1
        direction = 2
        obstacle_positions = 2 * len(
            FullStateDataExtractor.get_obstacle_positions(self.init_full_state)
        )
        partial_state_size = (
            agent_position + goal_position + distance + direction + obstacle_positions
        )
        return np.array([partial_state_size], dtype=np.uint8)

    @property
    def full_state_size(self) -> np.ndarray:
        return np.array([self.height, self.width], dtype=np.uint8)
