"""Tag environment for reinforcement learning."""

__credits__ = ["Eirik Reiestad"]

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments import settings
from environments.gymnasium.utils import (
    FileHandler,
    Position,
    generate_random_position,
)

from .tag_renderer import TagRenderer
from .tag_rewards import TagRewards
from .tag_state import TagState
from .env_utils import EnvUtils
from .utils import (
    AGENT_TILE_TYPE,
    Agent,
    AgentType,
    DualAgents,
    TileType,
    ActionType,
    Objects,
    Object,
    ObjectType,
)

logging.basicConfig(level=logging.INFO)


class TagEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = "human"):
        self.height = settings.ENV_HEIGHT
        self.width = settings.ENV_WIDTH
        self.max_steps = self.height * self.width

        folder_name = "environments/gymnasium/data/tag/"
        FileHandler.file_exist(folder_name, settings.FILENAME)

        self.tag_renderer = TagRenderer(
            settings.ENV_HEIGHT,
            settings.ENV_WIDTH,
            settings.SCREEN_WIDTH,
            settings.SCREEN_HEIGHT,
        )
        self.tag_renderer.init_render_mode(render_mode)

        filename = folder_name + settings.FILENAME
        self.state = TagState(self.height, self.width, filename)
        self._init_spaces()
        self.tag_rewards = TagRewards()

        self.steps = 0
        self.steps_beyond_terminated = None
        self._set_initial_positions(None)

        self.info = {
            "object_moved_distance": 0,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self.steps += 1
        if self.steps >= self.max_steps:
            return (
                self.state.active_state,
                self.tag_rewards.truncated_reward,
                True,
                True,
                {"full_state": self.state.full},
            )

        collided = False
        terminated = False
        reward = 0

        if not terminated:
            action_type = ActionType(action)
            new_full_state, reward = self._do_action(action_type)
            collided = new_full_state is None
            if not collided and new_full_state is not None:
                self.state.update(
                    new_full_state,
                    self.agents.active.position,
                    self.agents.inactive.position,
                    self.objects,
                )
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
            "object_moved_distance": self.info["object_moved_distance"],
        }

        return (self.state.active_state, reward, terminated, False, return_info)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if options is not None and options.get("all_possible_states"):
            return self.state.active_state, {
                "state_type": settings.STATE_TYPE.value,
                "all_possible_states": self.state.get_all_possible_states(
                    self.agents.active_agent, self.agents.inactive_agent, self.objects
                ),
            }
        super().reset(seed=seed)
        render_mode = options.get("render_mode") if options else None
        self.render_mode = render_mode or self.render_mode

        self.info["object_moved_distance"] = 0

        self.state.reset(self.agents.active_agent)
        self._set_initial_positions(options)
        self._set_object_positions()

        self.steps = 0
        self.steps_beyond_terminated = None

        return self.state.active_state, {"state_type": settings.STATE_TYPE.value}

    def render(self, render_mode: Optional[str] = None) -> Optional[np.ndarray]:
        return self.tag_renderer.render(self.state.full, render_mode)

    def close(self):
        self.tag_renderer.close()

    def set_active_agent(self, agent: AgentType):
        self.agents.active_agent = agent

    def concatenate_states(
        self, states: list[np.ndarray]
    ) -> tuple[np.ndarray, tuple[float, float], bool]:
        state, _ = self.state.concatenate_states(states)
        rewards, terminated = self.tag_rewards.get_tag_reward(
            self.agents.active.position,
            self.agents.inactive.position,
            settings.TAG_RADIUS,
        )
        return state, rewards, terminated

    def update_state(self, state: np.ndarray) -> np.ndarray:
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
        if action == ActionType.GRAB:
            reward += self._grab_entity()
        if action == ActionType.RELEASE:
            reward += self._release_entity()
        new_full_state, move_reward = self._move_agent(self.state.full, action)
        reward += move_reward
        if new_full_state is None:
            return None, reward
        new_full_state = self._move_grabbed_object(new_full_state)
        return new_full_state, reward

    def _grab_entity(self) -> float:
        for obj in self.objects.boxes:
            if self.agents.active.position.distance_to(obj.position) <= 1:
                can_grab = obj.grab(self.agents.active_agent)
                if can_grab:
                    obj.grabbed = True
                    obj.next_position = self.agents.active.position
                    self.agents.active.grab(obj)
                    return 0
                return self.tag_rewards.wrong_grab_reward
        return self.tag_rewards.wrong_grab_reward

    def _release_entity(self):
        ok = self.agents.active.release()
        return 0 if ok else self.tag_rewards.wrong_release_reward

    def _move_grabbed_object(self, state: np.ndarray) -> Optional[np.ndarray]:
        obj = self.agents.active.grabbed_object
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
                self.info["object_moved_distance"] += 1
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
        if EnvUtils.is_within_bounds(new_state, new_agent_position):
            if not EnvUtils.is_object(new_state, new_agent_position):
                return self._move_agent_within_bounds(
                    new_state, new_agent_position
                ), self.tag_rewards.move_reward
            else:
                return state, self.tag_rewards.collision_reward
        else:
            return None, self.tag_rewards.terminated_reward

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

    def _set_initial_positions(self, options: Optional[Dict[str, Any]]):
        """Sets the initial positions of the agent and goal."""
        if options and "seeker" in options:
            seeker_position = Position(options["hider"])
            hider_position = Position(
                options.get(
                    "hider",
                    generate_random_position(
                        self.width, self.height, [seeker_position]
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
        self.agents = DualAgents(seeker, hider, AgentType.SEEKER)

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

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return 5
