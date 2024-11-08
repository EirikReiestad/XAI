import logging
import os

import numpy as np
from environments.gymnasium.envs.tag.utils import (
    AgentType,
    FullStateDataExtractor,
    FullStateDataModifier,
    TileType,
    Objects,
    Object,
    ObjectType,
)
from rl.src.common.getter import get_torch_from_numpy
from environments.gymnasium.utils import Position, State, StateType
from utils import Color
from typing import Callable


class TagState:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        state_type: StateType,
        filename: str,
    ):
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._state_type = state_type
        self._random_seeker_position = False
        self._random_hider_position = False
        self._random_box_position = False
        self._init_states(filename)
        self._init_dimensions()

    def reset(self):
        self._state.full = self.init_full_state
        seeker_position = FullStateDataExtractor.get_agent_position(
            self._state.full, AgentType.SEEKER
        )
        hider_position = FullStateDataExtractor.get_agent_position(
            self._state.full, AgentType.HIDER
        )
        obstacle_positions = FullStateDataExtractor.get_positions(
            self._state.full, TileType.OBSTACLE
        )
        box_positions = FullStateDataExtractor.get_positions(
            self._state.full, TileType.BOX
        )
        powerup0_positions = FullStateDataExtractor.get_positions(
            self._state.full, TileType.POWERUP0
        )
        powerup1_positions = FullStateDataExtractor.get_positions(
            self._state.full, TileType.POWERUP1
        )
        obstacles = [
            Object(ObjectType.OBSTACLE, position, False)
            for position in obstacle_positions
        ]
        boxes = [Object(ObjectType.BOX, position, True) for position in box_positions]
        powerup0 = [
            Object(ObjectType.POWERUP0, position, True)
            for position in powerup0_positions
        ]
        powerup1 = [
            Object(ObjectType.POWERUP1, position, True)
            for position in powerup1_positions
        ]
        objects = Objects(obstacles, boxes, powerup0, powerup1)

        self.validate_state(self._state.full)

        self._state.partial = self._create_partial_state(
            seeker_position, hider_position, objects
        )
        self._state.rgb = self._create_rgb_state()

    def update(
        self,
        new_full_state: np.ndarray,
        seeker_position: Position,
        hider_position: Position,
        objects: Objects,
    ):
        self._state.full = new_full_state
        self._state.partial = self._create_partial_state(
            seeker_position, hider_position, objects
        )
        self._state.rgb = self._create_rgb_state()

    def get_agent_position(
        self, agent: AgentType, state: np.ndarray | None = None
    ) -> Position:
        if state is None:
            return FullStateDataExtractor.get_agent_position(self._state.full, agent)
        return FullStateDataExtractor.get_agent_position(state, agent)

    def get_initial_agent_position(self, agent: AgentType) -> Position:
        assert self.init_full_state is not None, "Initial full state is not set yet."
        return FullStateDataExtractor.get_agent_position(self.init_full_state, agent)

    def concatenate_states(self, states: list[np.ndarray]) -> tuple[np.ndarray, bool]:
        if len(states) == 1:
            return self._concatenate_single_state(states[0])
        elif len(states) == 2:
            return self._concatenate_double_state(states)
        else:
            raise ValueError("Two states are required to concatenate.")

    def get_obstacle_positions(self) -> list[Position]:
        return FullStateDataExtractor.get_positions(self.full, TileType.OBSTACLE)

    def get_box_positions(self) -> list[Position]:
        return FullStateDataExtractor.get_positions(self.full, TileType.BOX)

    def get_powerup_positions(self, num: int) -> list[Position]:
        if num == 0:
            return FullStateDataExtractor.get_positions(self.full, TileType.POWERUP0)
        if num == 1:
            return FullStateDataExtractor.get_positions(self.full, TileType.POWERUP1)
        raise ValueError(f"Powerup number {num} is not implemented yet.")

    def get_all_possible_states(
        self, active_agent: AgentType, inactive_agent: AgentType, objects: Objects
    ) -> np.ndarray:
        if self._state_type == StateType.FULL:
            return self._get_all_possible_full_states(active_agent)
        elif self._state_type == StateType.PARTIAL:
            return self._get_all_possible_partial_states(
                active_agent, inactive_agent, objects
            )
        elif self._state_type == StateType.RGB:
            raise NotImplementedError("RGB state type not implemented yet.")
        raise ValueError(f"Unknown state type: {self._state_type}")

    def validate_state(self, state: np.ndarray):
        FullStateDataExtractor.get_agent_position(state, AgentType.SEEKER)
        FullStateDataExtractor.get_agent_position(state, AgentType.HIDER)

    def get_occluded_states(self) -> np.ndarray:
        assert (
            self._state_type == StateType.FULL
        ), f"Occlusion is only supported for full state type, not {self._state_type}"

        states = np.ndarray(
            (self.height, self.width, *self.full_state_size), dtype=np.uint8
        )

        for y in range(self.height):
            for x in range(self.width):
                state = self._state.full.copy()
                state = FullStateDataModifier.occlude(state, Position(x, y))
                torch_state = get_torch_from_numpy(state)
                states[y, x] = torch_state.unsqueeze(0)
        return states

    def place_seeker_next_to_hider(self):
        new_full_state = FullStateDataModifier.place_seeker_next_to_hider(
            self._state.full
        )
        self.update(
            new_full_state,
            self.get_agent_position(AgentType.SEEKER),
            self.get_agent_position(AgentType.HIDER),
            Objects([], [], [], []),
        )

    def place_agent_next_to_box(self, agent_type: AgentType):
        new_full_state = FullStateDataModifier.place_agent_next_to_box(
            self._state.full, agent_type
        )
        self.update(
            new_full_state,
            self.get_agent_position(AgentType.SEEKER),
            self.get_agent_position(AgentType.HIDER),
            Objects([], [], [], []),
        )

    def remove_box(self):
        new_full_state, _ = FullStateDataModifier.remove_objects(
            self._state.full, TileType.BOX
        )
        self.update(
            new_full_state,
            self.get_agent_position(AgentType.SEEKER),
            self.get_agent_position(AgentType.HIDER),
            Objects([], [], [], []),
        )

    def remove_agent(self, agent_type: AgentType):
        new_full_state = FullStateDataModifier.remove_agent(
            self._state.full, agent_type
        )
        if agent_type == AgentType.SEEKER:
            return self.update(
                new_full_state,
                Position(-1, -1),
                self.get_agent_position(AgentType.HIDER),
                Objects([], [], [], []),
            )
        return self.update(
            new_full_state,
            self.get_agent_position(AgentType.SEEKER),
            Position(-1, -1),
            Objects([], [], [], []),
        )

    def has_direct_sight(self, state: np.ndarray) -> tuple[bool, list[Position]]:
        return FullStateDataExtractor.has_direct_sight(state)

    def place_agent_with_direct_sight(self):
        new_full_state = self._get_random_state_with_criteria(
            lambda state: self.has_direct_sight(state)[0]
        )
        return self.update(
            new_full_state,
            self.get_agent_position(AgentType.SEEKER, new_full_state),
            self.get_agent_position(AgentType.HIDER, new_full_state),
            Objects([], [], [], []),
        )

    def place_agent_with_no_direct_sight(self):
        new_full_state = self._get_random_state_with_criteria(
            lambda state: not self.has_direct_sight(state)[0]
        )
        return self.update(
            new_full_state,
            self.get_agent_position(AgentType.SEEKER, new_full_state),
            self.get_agent_position(AgentType.HIDER, new_full_state),
            Objects([], [], [], []),
        )

    def place_agents_far_apart(self):
        radius = self.height // 2
        new_full_state = FullStateDataModifier.place_agents_far_apart(
            self._state.full, radius
        )
        self.update(
            new_full_state,
            self.get_agent_position(AgentType.SEEKER),
            self.get_agent_position(AgentType.HIDER),
            Objects([], [], [], []),
        )

    @property
    def init_full_state(self):
        init_state = self._init_full_state

        if self._random_seeker_position:
            init_state = FullStateDataModifier.random_agent_position(
                init_state, AgentType.SEEKER
            )
        if self._random_hider_position:
            init_state = FullStateDataModifier.random_agent_position(
                init_state, AgentType.HIDER
            )
        if self._random_box_position:
            init_state = FullStateDataModifier.random_objects_position(
                init_state, TileType.BOX
            )

        self.validate_state(init_state)
        return init_state

    @init_full_state.setter
    def init_full_state(self, value: np.ndarray):
        self._init_full_state = value

    @property
    def state_type(self) -> StateType:
        return self._state_type

    @property
    def active_state(self, normalized: bool = True) -> np.ndarray:
        if normalized:
            return self._state.normalized_full_state
        return self._state.active_state

    @property
    def normalized_full_state(self) -> np.ndarray:
        return self._state.normalized_full_state

    @property
    def full(self) -> np.ndarray:
        return self._state.full

    @full.setter
    def full(self, value: np.ndarray):
        self._state.full = value

    @property
    def random_seeker_position(self) -> bool:
        return self._random_seeker_position

    @random_seeker_position.setter
    def random_seeker_position(self, value: bool):
        self._random_seeker_position = value

    @property
    def random_hider_position(self) -> bool:
        return self._random_hider_position

    @random_hider_position.setter
    def random_hider_position(self, value: bool):
        self._random_hider_position = value

    @property
    def random_box_position(self) -> bool:
        return self._random_box_position

    @random_box_position.setter
    def random_box_position(self, value: bool):
        self._random_box_position = value

    @property
    def partial(self) -> np.ndarray:
        return self._state.partial

    @property
    def rgb(self) -> np.ndarray:
        return self._state.rgb

    @property
    def full_state_size(self) -> tuple[int, int]:
        return self.init_full_state.shape[0], self.init_full_state.shape[1]

    @property
    def partial_state_size(self) -> np.ndarray:
        partial_state_size = (
            self._active_agnet_position_size
            + self._inactive_agent_position_size
            + self._distance_size
            + self._direction_size
            + self._obstacle_state_size
            + self._box_state_size
        )
        return np.array([partial_state_size], dtype=np.uint8)

    @property
    def _active_agnet_position_size(self) -> int:
        return 2

    @property
    def _inactive_agent_position_size(self) -> int:
        return 2

    @property
    def _distance_size(self) -> int:
        return 1

    @property
    def _direction_size(self) -> int:
        return 2

    @property
    def _obstacle_state_size(self) -> int:
        return self._num_obstacles * Object.state_size_static()

    @property
    def _num_obstacles(self) -> int:
        return len(
            FullStateDataExtractor.get_positions(
                self.init_full_state, TileType.OBSTACLE
            )
        )

    @property
    def _box_state_size(self) -> int:
        return self._num_boxes * Object.state_size_static()

    @property
    def _num_boxes(self) -> int:
        return len(
            FullStateDataExtractor.get_positions(self.init_full_state, TileType.BOX)
        )

    @property
    def agent_distance(self) -> float:
        seeker_position = FullStateDataExtractor.get_agent_position(
            self._state.full, AgentType.SEEKER
        )
        hider_position = FullStateDataExtractor.get_agent_position(
            self._state.full, AgentType.HIDER
        )
        distance_vector = seeker_position - hider_position
        distance = np.linalg.norm(distance_vector.tuple)

        return float(distance)

    def _get_random_state_with_criteria(
        self, criteria: Callable[[np.ndarray], bool]
    ) -> np.ndarray:
        new_full_state = self._state.full.copy()

        while True:
            new_full_state = FullStateDataModifier.random_agent_position(
                new_full_state, AgentType.SEEKER
            )
            new_full_state = FullStateDataModifier.random_agent_position(
                new_full_state, AgentType.HIDER
            )
            if criteria(new_full_state):
                break

        return new_full_state

    def _init_states(self, filename: str):
        self.init_full_state = self._load_env_from_file(filename)
        full_state = self.init_full_state
        partial_state = np.ndarray(self.partial_state_size, dtype=np.uint8)
        rgb_state = self._create_rgb_state()

        self._state = State(
            full=full_state,
            partial=partial_state,
            rgb=rgb_state,
            active=self._state_type,
        )

    def _init_dimensions(self):
        self.width = self.init_full_state.shape[1]
        self.height = self.init_full_state.shape[0]

    def _load_env_from_file(self, filename: str) -> np.ndarray:
        if not os.path.exists(filename):
            logging.info(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"File {filename} does not exist.")

        with open(filename, "r") as f:
            env = [list(map(int, list(row.strip()))) for row in f.readlines()]

        return np.array(env, dtype=np.uint8)

    def _create_rgb_state(self) -> np.ndarray:
        return np.full((self._screen_height, self._screen_width, 3), Color.WHITE.value)

    def _create_partial_state(
        self, seeker_position: Position, hider_position: Position, objects: Objects
    ) -> np.ndarray:
        goal_distance = seeker_position - hider_position
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
                *hider_position,
                *seeker_position,
                int(distance_normalized),
                *map(int, direction_normalized),
                *object_states,
            ],
            dtype=np.uint8,
        )

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
        # TODO: This might be wrong logic (for box or moveable objects in general)? Should compare the last state?
        box_positions = FullStateDataExtractor.get_positions(hider_state, TileType.BOX)
        seeker_powerup0_positions = FullStateDataExtractor.get_positions(
            seeker_state, TileType.POWERUP0
        )
        hider_powerup0_positions = FullStateDataExtractor.get_positions(
            hider_state, TileType.POWERUP0
        )
        seeker_powerup1_positions = FullStateDataExtractor.get_positions(
            seeker_state, TileType.POWERUP1
        )
        hider_powerup1_positions = FullStateDataExtractor.get_positions(
            hider_state, TileType.POWERUP1
        )

        state = np.zeros((self.height, self.width), dtype=np.float32)
        for powerup0_position in seeker_powerup0_positions + hider_powerup0_positions:
            state[*powerup0_position.row_major_order] = TileType.POWERUP0.value
        for powerup1_position in seeker_powerup1_positions + hider_powerup1_positions:
            state[*powerup1_position.row_major_order] = TileType.POWERUP1.value
        state[*seeker_position.row_major_order] = TileType.SEEKER.value
        state[*hider_position.row_major_order] = TileType.HIDER.value
        for box_position in box_positions:
            state[*box_position.row_major_order] = TileType.BOX.value
        for obstacle_position in obstacle_positions:
            state[*obstacle_position.row_major_order] = TileType.OBSTACLE.value

        if seeker_position == hider_position:
            return state, True

        self.validate_state(state)
        return state, False

    def _get_all_possible_full_states(self, active_agent: AgentType) -> np.ndarray:
        clean_agent_state = self._state.full.copy()
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

        states = np.ndarray(
            (self.height, self.width, *self.partial_state_size), dtype=np.uint8
        )
        for y in range(self.height):
            for x in range(self.width):
                active_agent_position = Position(x, y)
                if FullStateDataExtractor.is_empty_tile(
                    clean_agent_state, active_agent_position
                ):
                    if active_agent == AgentType.SEEKER:
                        state = self._state.partial = self._create_partial_state(
                            active_agent_position, inactive_agent_position, objects
                        )
                    else:
                        state = self._state.partial = self._create_partial_state(
                            inactive_agent_position, active_agent_position, objects
                        )
                else:
                    state = self._create_empty_partial_state()
                states[active_agent_position.row_major_order] = state
        return states

    def _create_empty_full_state(self):
        return np.zeros((self.height, self.width), dtype=np.uint8)

    def _create_empty_partial_state(self):
        return np.ndarray(self.partial_state_size, dtype=np.uint8)
