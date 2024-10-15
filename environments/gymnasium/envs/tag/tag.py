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
<<<<<<< HEAD
        self.height = 10
        self.width = 10
        self.screen_width = 600
        self.screen_height = 600
=======
        screen_width = 600
        screen_height = 600
>>>>>>> 300c75a (feat: extract height and width from file)
        folder_name = "environments/gymnasium/data/tag/"
        filename = "tag-0-7-7.txt"
        self.state_type = StateType.FULL
        self.bootcamp = Bootcamp()
        self.tag_radius = 1
        self.tag_head_start = 0
        self.max_steps = 200
        self.freeze_hider = True
        self.terminate_out_of_bounds = False

        FileHandler.file_exist(folder_name, filename)

<<<<<<< HEAD
        self.tag_renderer = TagRenderer(
            self.width, self.height, self.screen_width, self.screen_height
        )
        self.render_mode = render_mode

        filename = folder_name + filename
        self.state = TagState(
            self.width,
            self.height,
            self.screen_width,
            self.screen_height,
            self.state_type,
            filename,
        )

=======
        filename = folder_name + filename
        self.state = TagState(
            screen_width,
            screen_height,
            self.state_type,
            filename,
        )
        self.tag_renderer = TagRenderer(
            self.state.width, self.state.height, screen_width, screen_height
        )
        self.render_mode = render_mode
>>>>>>> 300c75a (feat: extract height and width from file)
        self._init_spaces()
        self.tag_rewards = TagRewards()

        self.tag_concepts = TagConcepts(self.state, self.num_actions)

        self.steps = 0
        self.steps_beyond_terminated = None
        self._set_initial_positions(None)

        self.info = {
            "object_moved_distance": 0,
            "collided": 0,
            "wrong_grab_release": 0,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        self.steps += 1
        if (
            self.steps >= self.tag_head_start
            and self.freeze_hider
            and self.agents.active_agent == AgentType.HIDER
        ):
            self.bootcamp.step()
            return self._handle_agent_switch(True)
        if (
            (
                self.bootcamp.name in [BootcampName.HIDER]
                and self.agents.active_agent == AgentType.SEEKER
            )
            or (
                self.bootcamp.name in [BootcampName.SEEKER]
                and self.agents.active_agent == AgentType.HIDER
            )
            or (
                self.agents.active_agent == AgentType.HIDER
                and not self.bootcamp.move_hider(self.steps)
            )
            or (
                self.agents.active_agent == AgentType.SEEKER
                and not self.bootcamp.move_seeker(self.steps)
            )
            or (
                self.steps < self.tag_head_start
                and self.agents.active_agent == AgentType.SEEKER
            )
        ):
            return self._handle_agent_switch(True)

        if self.steps >= self.max_steps:
            reward = self.tag_rewards.terminated_reward
            return (
                self.state.active_state,
                reward,
                True,
                True,
                {
                    "full_state": self.state.full,
                    "data_additative": {
                        "object_moved_distance": self.info["object_moved_distance"],
                        "collided": self.info["collided"],
                        "wrong_grab_release": self.info["wrong_grab_release"],
                    },
                    "data_constant": {
                        "slow_factor": self.bootcamp.agent_slow_factor(
                            self.agents.active_agent
                        ),
                    },
                },
            )

        collided = False
        terminated = False
        reward = 0

        if not terminated:
            action_type = ActionType(action)
            self.info["collided"] = 0
            new_full_state, reward = self._do_action(action_type)
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

        return_info = {
            "full_state": self.state.full,
            "data_additative": {
                "object_moved_distance": self.info["object_moved_distance"],
                "collided": self.info["collided"],
                "wrong_grab_release": self.info["wrong_grab_release"],
            },
            "data_constant": {
                "slow_factor": self.bootcamp.agent_slow_factor(
                    self.agents.active_agent
                ),
            },
        }

        self.agents.set_next_agent()
        self.bootcamp.step()

        return (
            self.state.active_state,
            reward,
            terminated,
            False,
            return_info,
        )

    def _handle_agent_switch(
        self, skip: bool
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.agents.set_next_agent()
        return (
            self.state.active_state,
            0,
            False,
            False,
            {
                "full_state": self.state.full,
                "skip": skip,
                "data_additative": {
                    "object_moved_distance": self.info["object_moved_distance"],
                    "collided": self.info["collided"],
                    "wrong_grab_release": self.info["wrong_grab_release"],
                },
                "agent_slow_factor": self.bootcamp.agent_slow_factor(
                    self.agents.active_agent
                ),
            },
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        render_mode = options.get("render_mode") if options else None
        full_reset = options.get("full_reset") if options else False
        self.render_mode = render_mode or self.render_mode

        self.info = {"object_moved_distance": 0, "collided": 0, "wrong_grab_release": 0}

        self.state.reset()
        self._set_initial_positions(options)
        self._set_object_positions()

        self.tag_rewards.reset()

        self.steps = 0
        self.steps_beyond_terminated = None

        if full_reset:
            self.bootcamp.reset()

        return_state = self.state.active_state
        return return_state, {"state_type": self.state_type.value}

    def render(self, render_mode: Optional[str] = None) -> Optional[np.ndarray]:
        return self.tag_renderer.render(self.state.full, render_mode)

    def close(self):
        self.tag_renderer.close()

    def set_active_agent(self, agent: AgentType):
        self.agents.active_agent = agent

    def concatenate_states(
        self, states: list[np.ndarray]
    ) -> tuple[np.ndarray, tuple[float, float], bool, bool, int, int]:
        state, concat_terminated = self.state.concatenate_states(states)
        rewards, tag_terminated = self.tag_rewards.get_tag_reward(
            self.agents.active.position,
            self.agents.inactive.position,
            concat_terminated,
            self.tag_radius,
        )
        truncated = False
        if self.steps >= self.max_steps:
            rewards = self.tag_rewards.end_reward
            truncated = True
        self.render(self.render_mode)

        terminated = concat_terminated or tag_terminated

        seeker_won = 1 if terminated else 0
        hider_won = 0 if terminated else 1
        return state, rewards, terminated, truncated, seeker_won, hider_won

    def update_state(self, state: np.ndarray) -> np.ndarray:
        self.state.validate_state(state)
        self.state.update(
            state,
            self.agents.active.position,
            self.agents.inactive.position,
            self.objects,
        )
        return self.state.active_state

    def get_active_state(self) -> np.ndarray:
        return self.state.active_state

    def _do_action(self, action: ActionType) -> tuple[Optional[np.ndarray], float]:
        reward = 0
        if action == ActionType.GRAB_RELEASE:
            if self.agents.active.grabbed_object is not None:
                release = self._release_entity()
                if not release:
                    self.info["wrong_grab_release"] = 1
                    reward += self.tag_rewards.wrong_grab_release_reward
            else:
                grab = self._grab_entity()
                if not grab:
                    self.info["wrong_grab_release"] = 1
                    reward += self.tag_rewards.wrong_grab_release_reward
        new_full_state, move_reward = self._move_agent(self.state.full, action)
        reward += move_reward
        if new_full_state is None:
            return None, reward
        new_full_state = self._move_grabbed_object(new_full_state)
        if new_full_state is None:
            return None, reward
        return new_full_state, reward

    def _grab_entity(self) -> bool:
        for obj in self.objects.boxes:
            if self.agents.active.position.distance_to(obj.position) <= 1:
                can_grab = obj.can_grab()
                if can_grab:
                    obj.grabbed = True
                    obj.next_position = self.agents.active.position
                    self.agents.active.grab(obj)
                    return True
                return False
        return False

    def _release_entity(self):
        return self.agents.active.release()

    def _move_grabbed_object(self, state: np.ndarray) -> Optional[np.ndarray]:
        obj = self.agents.active.grabbed_object
        self.info["object_moved_distance"] = 0
        if obj is not None:
            if obj.next_position is None:
                raise ValueError("The object should have a next position.")
            next_position = self.agents.active.position
            if obj.next_position == next_position:
                return state

            new_obj_position = obj.next_position
            if EnvUtils.is_within_bounds(state, new_obj_position):
                state[*obj.position.row_major_order] = TileType.EMPTY.value
                obj.position = new_obj_position
                obj.next_position = self.agents.active.position
                state[*obj.position.row_major_order] = TileType.BOX.value
                self.info["object_moved_distance"] = 1
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
        new_agent_position = self.agents.active.position + action.direction.tuple
        if self.agents.inactive.position == new_agent_position:
            self.info["collided"] = 1
            return state, self.tag_rewards.collision_reward
        if EnvUtils.is_within_bounds(new_state, new_agent_position):
            if not EnvUtils.is_object(new_state, new_agent_position):
                return self._move_agent_within_bounds(
                    new_state, new_agent_position
                ), self.tag_rewards.move_reward[self.agents.active_agent.value]
            else:
                self.info["collided"] = 1
                return state, self.tag_rewards.collision_reward
        else:
            if self.terminate_out_of_bounds:
                return None, self.tag_rewards.terminated_reward
            return state, self.tag_rewards.collision_reward

    def _move_agent_within_bounds(
        self, state: np.ndarray, agent_position: Position
    ) -> Optional[np.ndarray]:
        state[*self.agents.active.position.row_major_order] = TileType.EMPTY.value
        self.agents.active.position = agent_position
        agent_tile_type = AGENT_TILE_TYPE.get(self.agents.active_agent)
        if agent_tile_type is None:
            raise ValueError(f"Invalid agent type {self.agents.active_agent}")
        state[*self.agents.active.position.row_major_order] = agent_tile_type
        return state

    def _init_spaces(self):
        """Initializes the action and observation spaces."""
        if self.state.active_state is None:
            raise ValueError("The state should be set before initializing spaces.")

        self.action_space = spaces.Discrete(self.num_actions)

        if self.state_type.value == "full":
            self.observation_space = spaces.Box(
                low=0, high=8, shape=self.state.full.shape, dtype=np.uint8
            )
        elif self.state_type.value == "partial":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=self.state.partial.shape, dtype=np.uint8
            )
        elif self.state_type.value == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=self.state.rgb.shape, dtype=np.uint8
            )
        else:
            raise ValueError(f"Invalid state type {self.state_type.value}")

    def _set_initial_positions(self, options: Optional[Dict[str, Any]]):
        """Sets the initial positions of the agent and goal."""
        if options and "seeker" in options:
            seeker_position = Position(options["hider"])
            hider_position = Position(
                options.get(
                    "hider",
                    generate_random_position(
                        self.state.width, self.state.height, [seeker_position]
                    ),
                )
            )
        else:
            seeker_position = self.state.get_agent_position(AgentType.SEEKER)
            hider_position = (
                self.state.get_agent_position(AgentType.HIDER)
                if options is None or "hider" not in options
                else Position(options["hider"])
            )
        seeker = Agent(seeker_position)
        hider = Agent(hider_position)
        self.agents = DualAgents(seeker, hider)

    def _set_object_positions(self):
        obstacle_positions = self.state.get_obstacle_positions()
        box_positions = self.state.get_box_positions()

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

    def get_all_possible_states(self) -> np.ndarray:
        return self.state.get_all_possible_states(
            self.agents.active_agent, self.agents.inactive_agent, self.objects
        )

    def get_occluded_states(self) -> np.ndarray:
        return self.state.get_occluded_states()

    def get_concept(
        self, concept: str, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        filename = "concept_env.txt"
        folder_name = "environments/gymnasium/data/tag/"
        FileHandler.file_exist(folder_name, filename)
        filename = folder_name + filename
        self.state = TagState(
            self.width,
            self.height,
            self.screen_width,
            self.screen_height,
            self.state_type,
            filename,
        )
        return self.tag_concepts.get_concept(concept, samples)

    @property
    def config(self) -> dict:
        rewards = self.tag_rewards.config
        return {
            "rewards": rewards,
        }

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def feature_names(self) -> list[str]:
        return self.state.feature_names

    @property
    def render_mode(self) -> str | None:
        return self.tag_renderer.render_mode

    @render_mode.setter
    def render_mode(self, mode: str | None):
        self.tag_renderer.render_mode = mode

    @property
    def concepts(self) -> dict:
        return self.tag_concepts.get_concepts_dict()

    @property
    def concept_names(self) -> list[str]:
        return self.tag_concepts.concept_names
