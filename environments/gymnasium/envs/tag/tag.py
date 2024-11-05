"""Tag environment for reinforcement learning."""

__credits__ = ["Eirik Reiestad"]

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.gymnasium.utils import (
    FileHandler,
    Position,
    StateType,
    generate_random_position,
)
from .env_utils import EnvUtils
from .tag_renderer import TagRenderer
from .tag_rewards import TagRewards
from .tag_state import TagState
from .tag_concepts import TagConcepts
from .utils import (
    AGENT_TILE_TYPE,
    ActionType,
    Agent,
    AgentType,
    Bootcamp,
    BootcampName,
    DualAgents,
    Object,
    Objects,
    ObjectType,
    TileType,
)

logging.basicConfig(level=logging.INFO)


class TagEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = "rgb_array"):
        self._height = 10
        self._width = 10
        self._screen_width = 600
        self._screen_height = 600
        self._state_type = StateType.FULL
        self._bootcamp = Bootcamp()
        self._tag_radius = 1
        self._tag_head_start = 0
        self._freeze_hider = False
        self._terminate_out_of_bounds = False
        self._max_steps = self._width * self._height * 4
        self._render_mode = render_mode
        self._steps = 0
        self._steps_beyond_terminated = None

        folder_name = "environments/gymnasium/data/tag/"
        filename = "maze-tag-0-10-10.txt"
        FileHandler.file_exist(folder_name, filename)
        filename = folder_name + filename

        self._state = TagState(
            self._screen_width,
            self._screen_height,
            self._state_type,
            filename,
        )
        self._tag_renderer = TagRenderer(
            self._state.width,
            self._state.height,
            self._screen_width,
            self._screen_height,
        )
        self._init_spaces()
        self._tag_rewards = TagRewards()
        self._tag_concepts = TagConcepts(self._state, self.num_actions)

        self._set_initial_positions(None)

        self._info = {
            "object_moved_distance": 0,
            "collided": 0,
            "wrong_grab_release": 0,
            "has_direct_sight": 0,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        if self._is_initial_switch_needed():
            self._bootcamp.step()
            return self._handle_agent_switch(False)
        self._steps += 1

        if self._is_switch_required():
            return self._handle_agent_switch(False)

        if self._steps >= self._max_steps:
            return self._end_episode()

        collided, terminated, reward = False, False, 0

        if not terminated:
            reward, collided = self._perform_action(action)
            terminated = terminated or collided

        if terminated:
            self._increment_termination_steps()
        else:
            self._steps_beyond_terminated = None

        return_info = self._generate_return_info()
        self._agents.set_next_agent()
        self._bootcamp.step()

        return (
            self._state.active_state,
            reward,
            terminated,
            False,
            return_info,
        )

    def set_state(self, state: np.ndarray) -> np.ndarray:
        return self.update_state(state)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._render_mode = options.get("render_mode") if options else self._render_mode

        self._info = {
            "object_moved_distance": 0,
            "collided": 0,
            "wrong_grab_release": 0,
            "has_direct_sight": 0,
        }

        self._state.reset()
        self._set_initial_positions(options)
        self._set_object_positions()
        self._tag_rewards.reset()
        self._steps, self._steps_beyond_terminated = 0, None
        self._bootcamp.slow_hider_factor = 2
        if options.get("full_reset", False) if options else False:
            self._bootcamp.reset()

        return self._state.active_state, {"state_type": self._state_type.value}

    def render(self, render_mode: Optional[str] = None) -> Optional[np.ndarray]:
        render_mode = render_mode or self._render_mode
        return self._tag_renderer.render(self._state.full, render_mode)

    def close(self):
        self._tag_renderer.close()

    def set_active_agent(self, agent: AgentType):
        self._agents.active_agent = agent

    def concatenate_states(
        self, states: list[np.ndarray]
    ) -> tuple[np.ndarray, tuple[float, float], bool, bool, int, int]:
        state, concat_terminated = self._state.concatenate_states(states)
        has_direct_sight, sight_positions = self._state.has_direct_sight(state)
        rewards, tag_terminated = self._tag_rewards.get_tag_reward(
            self._agents.active.position,
            self._agents.inactive.position,
            has_direct_sight,
            concat_terminated,
            self._tag_radius,
        )

        truncated = self._steps >= self._max_steps
        if truncated:
            rewards = self._tag_rewards.end_reward

        self._tag_renderer.direct_sight_positions = sight_positions
        self.render()

        terminated = concat_terminated or tag_terminated
        seeker_won, hider_won = (1, 0) if terminated else (0, 1)
        return state, rewards, terminated, truncated, seeker_won, hider_won

    def update_state(self, state: np.ndarray) -> np.ndarray:
        self._state.validate_state(state)
        self._state.update(
            state,
            self._agents.active.position,
            self._agents.inactive.position,
            self.objects,
        )
        return self._state.active_state

    def get_active_state(self) -> np.ndarray:
        return self._state.active_state

    def get_all_possible_states(self, agent: str | None = None) -> np.ndarray:
        if agent is not None and agent == "seeker":
            return self._state.get_all_possible_states(
                AgentType.SEEKER, AgentType.HIDER, self.objects
            )
        if agent is not None and agent == "hider":
            return self._state.get_all_possible_states(
                AgentType.HIDER, AgentType.SEEKER, self.objects
            )
        return self._state.get_all_possible_states(
            self._agents.active_agent, self._agents.inactive_agent, self.objects
        )

    def get_occluded_states(self) -> np.ndarray:
        return self._state.get_occluded_states()

    def get_concept(
        self, concept: str, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        return self._tag_concepts.get_concept(concept, samples)

    @property
    def config(self) -> dict:
        rewards = self._tag_rewards.config
        return {
            "rewards": rewards,
        }

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return 4

    @property
    def feature_names(self) -> list[str]:
        return self._state.feature_names

    @property
    def render_mode(self) -> str | None:
        return self._tag_renderer.render_mode

    @render_mode.setter
    def render_mode(self, mode: str | None):
        self._tag_renderer.render_mode = mode

    @property
    def concepts(self) -> str:
        return str(self._tag_concepts)

    @property
    def concept_names(self) -> list[str]:
        return self._tag_concepts.concept_names

    def _increment_termination_steps(self):
        if self._steps_beyond_terminated is None:
            self._steps_beyond_terminated = 0
        else:
            if self._steps_beyond_terminated == 0:
                logging.warning(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self._steps_beyond_terminated += 1

    def _perform_action(self, action: int) -> Tuple[float, bool]:
        action_type = ActionType(action)
        self._update_render_action(action_type)
        self._info["collided"] = 0
        new_full_state, reward = self._do_action(action_type)
        collided = new_full_state is None
        if not collided and new_full_state is not None:
            self.update_state(new_full_state)
            has_direct_sight, _ = self._state.has_direct_sight(new_full_state)
            self._info["has_direct_sight"] = 1 if has_direct_sight else 0
        return reward, collided

    def _end_episode(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = self._tag_rewards.terminated_reward
        return (
            self._state.active_state,
            reward,
            True,
            True,
            {
                "full_state": self._state.full,
                "data_additative": {
                    "object_moved_distance": self._info["object_moved_distance"],
                    "collided": self._info["collided"],
                    "wrong_grab_release": self._info["wrong_grab_release"],
                    "has_direct_sight": self._info["has_direct_sight"],
                },
                "data_constant": {
                    "slow_factor": self._bootcamp.agent_slow_factor(
                        self._agents.active_agent
                    ),
                },
            },
        )

    def _is_initial_switch_needed(self) -> bool:
        return (
            self._steps >= self._tag_head_start
            and self._freeze_hider
            and self._agents.active_agent == AgentType.HIDER
        )

    def _is_switch_required(self) -> bool:
        return (
            (
                self._bootcamp.name in [BootcampName.HIDER]
                and self._agents.active_agent == AgentType.SEEKER
            )
            or (
                self._bootcamp.name in [BootcampName.SEEKER]
                and self._agents.active_agent == AgentType.HIDER
            )
            or (
                self._agents.active_agent == AgentType.HIDER
                and not self._bootcamp.move_hider(self._steps)
            )
            or (
                self._agents.active_agent == AgentType.SEEKER
                and not self._bootcamp.move_seeker(self._steps)
            )
            or (
                self._steps < self._tag_head_start
                and self._agents.active_agent == AgentType.SEEKER
            )
        )

    def _update_render_action(self, action: ActionType):
        if action == ActionType.GRAB_RELEASE:
            return
        if self._agents.active_agent == AgentType.SEEKER:
            self._tag_renderer.seeker_action = action
        else:
            self._tag_renderer.hider_action = action

    def _handle_agent_switch(
        self, skip: bool
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._agents.set_next_agent()
        return (
            self._state.active_state,
            0,
            False,
            False,
            {
                "full_state": self._state.full,
                "skip": skip,
                "data_additative": {
                    "object_moved_distance": self._info["object_moved_distance"],
                    "collided": self._info["collided"],
                    "wrong_grab_release": self._info["wrong_grab_release"],
                    "has_direct_sight": self._info["has_direct_sight"],
                },
                "agent_slow_factor": self._bootcamp.agent_slow_factor(
                    self._agents.active_agent
                ),
            },
        )

    def _do_action(self, action: ActionType) -> tuple[Optional[np.ndarray], float]:
        reward = 0

        if action == ActionType.GRAB_RELEASE:
            if self._agents.active.grabbed_object is not None:
                if not self._release_entity():
                    self._info["wrong_grab_release"] = 1
                    reward += self._tag_rewards.wrong_grab_release_reward
            else:
                if not self._grab_entity():
                    self._info["wrong_grab_release"] = 1
                    reward += self._tag_rewards.wrong_grab_release_reward

        new_full_state, move_reward = self._move_agent(self._state.full, action)
        reward += move_reward

        if new_full_state is None:
            return None, reward
        return self._move_grabbed_object(new_full_state), reward

    def _grab_entity(self) -> bool:
        for obj in self.objects.boxes:
            if (
                self._agents.active.position.distance_to(obj.position) <= 1
                and obj.can_grab()
            ):
                obj.grabbed = True
                obj.next_position = self._agents.active.position
                self._agents.active.grab(obj)
                return True
        return False

    def _release_entity(self):
        return self._agents.active.release()

    def _move_grabbed_object(self, state: np.ndarray) -> Optional[np.ndarray]:
        obj = self._agents.active.grabbed_object
        self._info["object_moved_distance"] = 0

        if obj is None or obj.next_position is None:
            return state

        next_position = self._agents.active.position

        if obj.next_position == next_position:
            return state

        if EnvUtils.is_within_bounds(state, obj.next_position):
            state[*obj.position.row_major_order] = TileType.EMPTY.value
            obj.position = obj.next_position
            obj.next_position = self._agents.active.position
            state[*obj.position.row_major_order] = TileType.BOX.value
            self._info["object_moved_distance"] = 1
        else:
            logging.warning(
                f"Object {obj.position} is outside the bounds of the environment."
            )
            return None
        return state

    def _move_agent(
        self, state: np.ndarray, action: ActionType
    ) -> tuple[Optional[np.ndarray], float]:
        new_state = state.copy()
        new_agent_position = self._agents.active.position + action.direction.tuple

        if self._agents.inactive.position == new_agent_position:
            self._info["collided"] = 1
            return state, self._tag_rewards.collision_reward

        if not EnvUtils.is_within_bounds(new_state, new_agent_position):
            return (
                (None, self._tag_rewards.terminated_reward)
                if self._terminate_out_of_bounds
                else (state, self._tag_rewards.collision_reward)
            )
        if EnvUtils.is_obstacle(new_state, new_agent_position):
            self._info["collided"] = 1
            return state, self._tag_rewards.collision_reward

        if (
            EnvUtils.is_box(new_state, new_agent_position)
            and self._agents.active_agent == AgentType.HIDER
        ):
            self._bootcamp.slow_hider_factor = 1

        return self._move_agent_within_bounds(
            new_state, new_agent_position
        ), self._tag_rewards.move_reward[self._agents.active_agent.value]

    def _move_agent_within_bounds(
        self, state: np.ndarray, agent_position: Position
    ) -> Optional[np.ndarray]:
        previous_position = self._agents.active.position.row_major_order
        state[*previous_position] = TileType.EMPTY.value

        self._agents.active.position = agent_position
        agent_tile_type = AGENT_TILE_TYPE.get(self._agents.active_agent)

        if agent_tile_type is None:
            raise ValueError(f"Invalid agent type {self._agents.active_agent}")

        state[*self._agents.active.position.row_major_order] = agent_tile_type
        return state

    def _init_spaces(self):
        """Initializes the action and observation spaces."""
        if self._state.active_state is None:
            raise ValueError("The state should be set before initializing spaces.")

        self.action_space = spaces.Discrete(self.num_actions)

        if self._state_type.value == "full":
            self.observation_space = spaces.Box(
                low=0, high=8, shape=self._state.full.shape, dtype=np.float64
            )
        elif self._state_type.value == "partial":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=self._state.partial.shape, dtype=np.uint8
            )
        elif self._state_type.value == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=self._state.rgb.shape, dtype=np.uint8
            )
        else:
            raise ValueError(f"Invalid state type {self._state_type.value}")

    def _set_initial_positions(self, options: Optional[Dict[str, Any]]):
        """Sets the initial positions of the agent and goal."""
        if options and "seeker" in options:
            seeker_position = Position(options["hider"])
            hider_position = Position(
                options.get(
                    "hider",
                    generate_random_position(
                        self._state.width, self._state.height, [seeker_position]
                    ),
                )
            )
        else:
            seeker_position = self._state.get_agent_position(AgentType.SEEKER)
            hider_position = (
                self._state.get_agent_position(AgentType.HIDER)
                if options is None or "hider" not in options
                else Position(options["hider"])
            )
        seeker = Agent(seeker_position)
        hider = Agent(hider_position)
        self._agents = DualAgents(seeker, hider)

    def _set_object_positions(self):
        obstacle_positions = self._state.get_obstacle_positions()
        box_positions = self._state.get_box_positions()

        obstacle_objects: list[Object] = []
        box_objects: list[Object] = []

        for position in obstacle_positions:
            obstacle_objects.append(
                Object(
                    object_type=ObjectType.OBSTACLE,
                    position=position,
                    grabable=False,
                )
            )

        for position in box_positions:
            box_objects.append(
                Object(
                    object_type=ObjectType.BOX,
                    position=position,
                    grabable=True,
                )
            )
        self.objects = Objects(obstacle_objects, box_objects)

    def _generate_return_info(self) -> Dict[str, Any]:
        return {
            "full_state": self._state.full,
            "data_additative": {
                "object_moved_distance": self._info["object_moved_distance"],
                "collided": self._info["collided"],
                "wrong_grab_release": self._info["wrong_grab_release"],
                "has_direct_sight": self._info["has_direct_sight"],
            },
            "data_constant": {
                "slow_factor": self._bootcamp.agent_slow_factor(
                    self._agents.active_agent
                ),
            },
        }
