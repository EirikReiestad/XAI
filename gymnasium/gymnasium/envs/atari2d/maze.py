"""
Maze system
"""

__credits__ = ["Eirik Reiestad"]

import math
import random
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.atari2d import utils
from gymnasium.error import DependencyNotInstalled


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

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None):
        self.height = 10
        self.width = 10
        self.max_steps = 100
        self.agent: utils.Position = utils.Position(
            random.randint(0, self.width), random.randint(0, self.height))
        self.goal: utils.Position = utils.Position(
            random.randint(0, self.width), random.randint(0, self.height))

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.height, self.width), dtype=np.uint8
        )

        self.rewards = {
            "goal": 1,
        }

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = np.zeros((self.height, self.width))

        self.state[self.agent.y, self.agent.x] = 1
        self.state[self.goal.y, self.goal.x] = 2

        self.steps_beyond_terminated = None

    def step(self, action: int) -> Tuple[np.ndarray, int, bool]:
        assert self.action_space.contains(
            action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        if self.steps_beyond_terminated is not None:
            self.steps_beyond_terminated += 1
            if self.steps_beyond_terminated >= 1:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."
                )

        self.state = move_agent(self.state, action)

        reward = 0
        if self.agent == self.goal:
            reward = self.rewards["goal"]
        terminated = self.agent == self.goal or self.steps_beyond_terminated >= 1

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, {}

    def reset(self) -> np.ndarray:
        self.agent = utils.Position(
            random.randint(0, self.width), random.randint(0, self.height))
        self.goal = utils.Position(
            random.randint(0, self.width), random.randint(0, self.height))
        self.state = np.zeros((self.height, self.width))
        self.state[self.agent.y, self.agent.x] = 1
        self.state[self.goal.y, self.goal.x] = 2
        self.steps_beyond_terminated = None
        return self.state

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "No render mode was set. Using 'human' as default.")
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "To use the render mode you need to install the 'pygame' package."
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        agent_size = min(self.screen_width, self.screen_height) // max(
            self.width, self.height)

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.surfarray.make_surface(x)
        self.surf.fill((255, 255, 255))

        agent_rect = pygame.Rect(
            self.agent.x, self.agent.y, agent_size, agent_size)
        gfxdraw.box(self.surf, agent_rect, (0, 0, 0))
        gfxdraw.box()

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def move_agent(state: (np.ndarray, Position), action: int) -> list[np.ndarray, utils.Position]:
    """
    Move the agent in the maze.
    :Arguments
    - state: The current state of the maze.
    - action: The action to take.
    :Returns
    - The new state of the maze.
    - A boolean indicating if the agent reached the goal.
    """

    maze, agent = state

    new_state = state.copy()
    new_agent = agent + utils.DIRECTIONS[action]
    if new_agent.x >= 0 and new_agent.x < self.width and new_agent.y >= 0 and new_agent.y < self.height:
        self.agent = new_agent
    return new_state
