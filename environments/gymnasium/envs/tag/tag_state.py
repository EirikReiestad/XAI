import logging
import os

import numpy as np
from environments import settings
from environments.gymnasium.envs.tag.utils import (
    AgentType,
    FullStateDataExtractor,
    FullStateDataModifier,
    TileType,
    Objects,
    Object,
    ObjectType,
)
from environments.gymnasium.utils import Position, State, StateType
from utils import Color


class TagState:
    def __init__(self, height: int, width: int, filename: str):
        self.height = height
        self.width = width
        self.init_states(filename)

    def init_states(self, filename: str):
        self.init_full_state = self._load_env_from_file(filename)
        full_state = self.init_full_state
        partial_state = np.ndarray(self.partial_state_size, dtype=np.uint8)
        rgb_state = self._create_rgb_state()

        self.state = State(
            full=full_state,
            partial=partial_state,
            rgb=rgb_state,
            active=settings.STATE_TYPE,
        )

    def reset(self, active_agent: AgentType):
        self.state.full = self.init_full_state
        active_agent_position = FullStateDataExtractor.get_agent_position(
            self.init_full_state, active_agent
        )
        other_agent = (
            AgentType.HIDER if active_agent == AgentType.SEEKER else AgentType.SEEKER
        )
        other_agent_position = FullStateDataExtractor.get_agent_position(
            self.init_full_state, other_agent
        )
        obstacle_positions = FullStateDataExtractor.get_positions(
            self.init_full_state, TileType.OBSTACLE
        )
        box_positions = FullStateDataExtractor.get_positions(
            self.init_full_state, TileType.BOX
        )
        obstacles = [
            Object(ObjectType.OBSTACLE, position, False)
            for position in obstacle_positions
        ]
        boxes = [Object(ObjectType.BOX, position, True) for position in box_positions]
        objects = Objects(obstacles, boxes)

        self.state.partial = self._create_partial_state(
            active_agent_position, other_agent_position, objects
        )
        self.state.rgb = self._create_rgb_state()

    def update(
        self,
        new_full_state: np.ndarray,
        active_agent_position: Position,
        other_agent_position: Position,
        objects: Objects,
    ):
        self.state.full = new_full_state
        self.state.partial = self._create_partial_state(
            active_agent_position, other_agent_position, objects
        )
        self.state.rgb = self._create_rgb_state()

    def get_agent_position(self, agent: AgentType) -> Position:
        return FullStateDataExtractor.get_agent_position(self.state.full, agent)

    def get_initial_agent_position(self, agent: AgentType) -> Position:
        if self.init_full_state is None:
            raise ValueError("Initial full state is not set yet.")
        return FullStateDataExtractor.get_agent_position(self.init_full_state, agent)

    def concatenate_states(self, states: list[np.ndarray]) -> tuple[np.ndarray, bool]:
        if len(states) == 1:
            return self._concatenate_single_state(states[0])
        elif len(states) == 2:
            return self._concatenate_double_state(states)
        else:
            raise ValueError("Two states are required to concatenate.")

    def _concatenate_single_state(self, state: np.ndarray) -> tuple[np.ndarray, bool]:
        seeker_exists = FullStateDataExtractor.agent_exist(state, AgentType.SEEKER)
        hider_exists = FullStateDataExtractor.agent_exist(state, AgentType.HIDER)

        if seeker_exists and hider_exists:
            seeker_position = FullStateDataExtractor.get_agent_position(
                state, AgentType.SEEKER
            )
            hider_position = FullStateDataExtractor.get_agent_position(
                state, AgentType.HIDER
            )
            return state, seeker_position == hider_position
        return state, False

    def _concatenate_double_state(
        self, states: list[np.ndarray]
    ) -> tuple[np.ndarray, bool]:
        seeker_state = states[0]
        hider_state = states[1]

        seeker_position = FullStateDataExtractor.get_agent_position(
            seeker_state, AgentType.SEEKER
        )
        hider_position = FullStateDataExtractor.get_agent_position(
            hider_state, AgentType.HIDER
        )
        obstacle_positions = FullStateDataExtractor.get_positions(
            seeker_state, TileType.OBSTACLE
        )
        hider_box_positions = FullStateDataExtractor.get_positions(
            hider_state, TileType.BOX
        )

        state = np.zeros((self.height, self.width), dtype=np.float32)
        state[*seeker_position.row_major_order] = TileType.SEEKER.value
        state[*hider_position.row_major_order] = TileType.HIDER.value
        for obstacle_position in obstacle_positions:
            state[*obstacle_position.row_major_order] = TileType.OBSTACLE.value
        for box_position in hider_box_positions:
            state[*box_position.row_major_order] = TileType.BOX.value

        if seeker_position == hider_position:
            return state, True

        self._validate_state(state)
        return state, False

    def get_obstacle_positions(self) -> list[Position]:
        return FullStateDataExtractor.get_positions(self.full, TileType.OBSTACLE)

    def get_box_positions(self) -> list[Position]:
        return FullStateDataExtractor.get_positions(self.full, TileType.BOX)

    def _validate_state(self, state: np.ndarray):
        FullStateDataExtractor.get_agent_position(state, AgentType.SEEKER)
        FullStateDataExtractor.get_agent_position(state, AgentType.HIDER)

    def _create_partial_state(
        self,
        active_agent_position: Position,
        other_agent_position: Position,
        objects: Objects,
    ) -> np.ndarray:
        goal_distance = active_agent_position - other_agent_position
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

        object_states = objects.state

        return np.array(
            [
                *active_agent_position,
                *other_agent_position,
                int(distance_normalized),
                *map(int, direction_normalized),
                *object_states,
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
            env = [list(map(int, list(row.strip()))) for row in f.readlines()]

        return np.array(env, dtype=np.uint8)

    @property
    def init_full_state(self):
        state = self._init_full_state

        def random_agent_position(state: np.ndarray, agent: AgentType) -> np.ndarray:
            state = FullStateDataModifier.remove_agent(self._init_full_state, agent)
            random_position = FullStateDataExtractor.get_random_position(
                state, TileType.EMPTY
            )
            return FullStateDataModifier.place_agent(state, random_position, agent)

        if settings.RANDOM_SEEKER_POSITION:
            random_agent_position(state, AgentType.SEEKER)
        if settings.RANDOM_HIDER_POSITION:
            random_agent_position(state, AgentType.HIDER)
        FullStateDataExtractor.get_agent_position(state, AgentType.SEEKER)
        FullStateDataExtractor.get_agent_position(state, AgentType.HIDER)
        return state

    @init_full_state.setter
    def init_full_state(self, value: np.ndarray):
        self._init_full_state = value

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

    def get_all_possible_states(
        self, active_agent: AgentType, inactive_agent: AgentType, objects: Objects
    ) -> np.ndarray:
        if settings.STATE_TYPE == StateType.FULL:
            return self._get_all_possible_full_states(active_agent)
        elif settings.STATE_TYPE == StateType.PARTIAL:
            return self._get_all_possible_partial_states(
                active_agent, inactive_agent, objects
            )
        elif settings.STATE_TYPE == StateType.RGB:
            raise NotImplementedError("RGB state type not implemented yet.")
        raise ValueError(f"Unknown state type: {settings.STATE_TYPE}")

    def _get_all_possible_full_states(self, active_agent: AgentType) -> np.ndarray:
        clean_agent_state = self.state.full.copy()
        clean_agent_state = FullStateDataModifier.remove_agent(
            clean_agent_state, active_agent
        )
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
                    state = FullStateDataModifier.place_agent(
                        state, agent_position, active_agent
                    )
                else:
                    state = self._create_empty_full_state()
                states[agent_position.row_major_order] = state
        return states

    def _get_all_possible_partial_states(
        self, active_agent: AgentType, inactive_agent: AgentType, objects: Objects
    ) -> np.ndarray:
        clean_agent_state = self.init_full_state.copy()
        clean_agent_state = FullStateDataModifier.remove_agent(
            clean_agent_state, active_agent
        )
        inactive_agent_position = FullStateDataExtractor.get_agent_position(
            self.init_full_state, inactive_agent
        )
        obstacle_positions = [obj.position for obj in objects.obstacles]
        box_positions = [obj.position for obj in objects.boxes]

        states = np.ndarray(
            (self.height, self.width, *self.partial_state_size), dtype=np.uint8
        )
        for y in range(self.height):
            for x in range(self.width):
                active_agent_position = Position(x, y)
                if FullStateDataExtractor.is_empty_tile(
                    clean_agent_state, active_agent_position
                ):
                    state = self.state.partial = self._create_partial_state(
                        active_agent_position, inactive_agent_position, objects
                    )
                else:
                    state = self._create_empty_partial_state()
                states[active_agent_position.row_major_order] = state
        return states

    @property
    def full_state_size(self) -> tuple[int, int]:
        return self.init_full_state.shape[0], self.init_full_state.shape[1]

    @property
    def partial_state_size(self) -> np.ndarray:
        active_agent_position = 2
        inactive_agent_position = 2
        distance = 1
        direction = 2
        num_obstacles = len(
            FullStateDataExtractor.get_positions(
                self.init_full_state, TileType.OBSTACLE
            )
        )
        num_boxes = len(
            FullStateDataExtractor.get_positions(self.init_full_state, TileType.BOX)
        )
        obstacle_state_size = num_obstacles * Object.state_size_static()
        box_state_size = num_boxes * Object.state_size_static()
        partial_state_size = (
            active_agent_position
            + inactive_agent_position
            + distance
            + direction
            + obstacle_state_size
            + box_state_size
        )
        return np.array([partial_state_size], dtype=np.uint8)

    def _create_empty_full_state(self):
        return np.zeros((self.height, self.width), dtype=np.uint8)

    def _create_empty_partial_state(self):
        return np.ndarray(self.partial_state_size, dtype=np.uint8)
