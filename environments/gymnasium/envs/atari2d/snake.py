"""
Snake system
"""

__credits__ = ["Eirik Reiestad"]

from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import spaces
import gymnasium.logger as logger
from gymnasium.error import DependencyNotInstalled

from environments.gymnasium.utils import Color

from . import utils


class SnakeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This class implements the snake game environment. The goal is to eat the food and grow the snake.

    ## Action Space
    The action is a ´ndarray´ with (1,) which can take values `{0, 1, 2}` indicating nothing, left or right.

    ## Observation Space

    The observation is a `ndarray` with shape `(height, width)` with the current state of the snake game.

    ## Rewards

    The reward is as follows:
    - `1` if the snake eats the food
    - `-1` if the snake dies

    ## Starting State

    The starting state is at a position in the middle of the board.
    The food is placed at a random position.

    ## Episode Termination

    The episode ends if any one of the following occurs:

    1. The snake dies.
    2. The snake reaches the maximum number of steps without progress (eating food).

    ## Arguments

    Optional arguments
    - height: The height of the grid.
    - width: The width of the grid.
    - max_steps: The maximum number of steps.
    - render_mode: The render mode.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None):
        self.height = 10
        self.width = 10
        self.max_steps = 100

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.height, self.width), dtype=np.uint8)

        self.rewards = {
            "food": 10,
            "step": 0,
            "terminate": -1,
        }

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = np.zeros((self.height, self.width), dtype=np.uint8)

        self.steps = 0
        self.last_snake_length = 0
        self.steps_beyond_terminated = None

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, dict]:
        """
        This method is called to take a step in the environment.
        Parameters:
            action: The action to take.

        Returns:
            np.ndarray: The new state of the environment.
            int: The reward of the action.
            bool: A boolean indicating if the episode is terminated.
            bool: A boolean indicating if the episode is truncated.
            dict: Additional information.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        if self.state is None:
            raise ValueError("Call reset before using step method.")

        self.steps += 1

        if self.steps >= self.max_steps:
            return self.state, 0, True, True, {}

        head = self.snake[0]

        if action == 0:
            pass
        elif action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4
        else:
            raise ValueError(f"Invalid action {action}")

        new_head = head + utils.DIRECTIONS[self.direction]

        terminated = new_head in self.snake[2:] or not (  # self.snake[2:] is the body of the snake
            0 <= new_head.x < self.width) or not (0 <= new_head.y < self.height)
        reward = self.rewards["terminate"] if terminated else self.rewards["step"]

        if not terminated:
            self.snake.insert(0, new_head)
            if head == self.food:
                self.steps = 0  # Reset steps because the snake ate the food
                self.food = utils.generate_random_position(
                    self.width-1, self.height-1, self.snake)
            else:
                self.snake.pop()
            reward = self.rewards["food"]
            self.state = np.zeros((self.height, self.width), dtype=np.uint8)
            for segment in self.snake:
                self.state[segment.y, segment.x] = 1
            self.state[self.food.y, self.food.x] = 2
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

        return self.state, reward, terminated, False, {}

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> np.ndarray:

        self.snake = [utils.Position(4, 4), utils.Position(
            3, 4), utils.Position(2, 4)]
        self.food = utils.generate_random_position(
            self.width-1, self.height-1, self.snake)
        self.direction = 1  # Rigth

        self.steps = 0
        self.state = np.zeros((self.height, self.width), dtype=np.uint8)

        for segment in self.snake:
            self.state[segment.y, segment.x] = 1

        self.state[self.food.y, self.food.x] = 2

        self.steps_beyond_terminated = None

        return self.state, {}

    def close(self) -> None:
        """
        This method is called to close the environment.
        """
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "No render mode specified. Using the default render mode 'human'.")
            return

        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "To use the render mode 'human', you need to install pygame. "
                "You can install it with `pip install pygame`."
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height))
                pygame.display.set_caption("Snake")
            else:
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.state is None:
            return None
        color_matrix = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8)
        snake_mask = self.state == 1
        food_mask = self.state == 2

        color_matrix[snake_mask] = Color.GREEN.value
        color_matrix[food_mask] = Color.RED.value

        surf = pygame.surfarray.make_surface(color_matrix)
        surf = pygame.transform.scale(
            surf, (self.screen_width, self.screen_height))
        surf = pygame.transform.flip(surf, True, False)

        self.screen.blit(surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(surf), axes=(1, 0, 2))
