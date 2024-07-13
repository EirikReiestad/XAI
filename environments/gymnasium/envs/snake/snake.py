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


class SnakeEnv(gym.Env[np.array, Union[int, np.ndarray]]):
    """
    ## Description

    This class implements the snake game environment. The goal is to eat the food and grow the snake.

    ## Action Space
    The action is a ´ndarray´ with (1,) which can take values `{0, 1, 2}` indicating nothing, left or right.

    ## Observation Space

    The state that is returned includes a tuple with the following elements:
    1. Relative Food Position: The position of the food relative to the head of the snake.
    2. Direction of Movement: Include the current direction of the snake's head as an integer. (´0´ for up, ´1´ for right, ´2´ for down, ´3´ for left)
    3. Immediate Obstacles: A list of 3 elements indicating if there is an obstacle in the direction of movement, to the left and to the right.
    4. Snake's Body: A list of positions of the snake's body.

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
        self.max_steps = self.height * self.width * 2

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(10,), dtype=np.float32)

        self.rewards = {
            "food": 1,
            "step": 0,
            "terminate": -1,
        }

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.raw_state = None

        self.steps = 0
        self.last_snake_length = 0
        self.steps_beyond_terminated = None

    def step(self, action: int) -> np.array:
        """
        This method is called to take a step in the environment.
        Parameters:
            action: The action to take.

        Returns:
            tuple: The new state of the environment.
            int: The reward of the action.
            bool: A boolean indicating if the episode is terminated.
            bool: A boolean indicating if the episode is truncated. dict: Additional information. """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        if self.raw_state is None:
            raise ValueError("Call reset before using step method.")

        self.steps += 1

        if self.steps >= self.max_steps:
            return self._get_state(), self.rewards["terminate"], True, True, {}

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
            self.raw_state = np.zeros(
                (self.height, self.width), dtype=np.uint8)
            for segment in self.snake:
                self.raw_state[segment.y, segment.x] = 1
            self.raw_state[self.food.y, self.food.x] = 2
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

        return self._get_state(), reward, terminated, False, {}

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.array, dict]:
        """
        The state that is returned includes a tuple with the following elements:
        1. Relative Food Position: The position of the food relative to the head of the snake.
        2. Direction of Movement: Include the current direction of the snake's head as an integer. (´0´ for up, ´1´ for right, ´2´ for down, ´3´ for left)
        3. Immediate Obstacles: A list of 3 elements indicating if there is an obstacle in the direction of movement, to the left and to the right.
        4. Snake's Body: A list of positions of the snake's body.
        """

        self.snake = [utils.Position(4, 4), utils.Position(
            3, 4), utils.Position(2, 4)]
        self.food = utils.generate_random_position(
            self.width-1, self.height-1, self.snake)
        self.direction = 1  # Rigth

        self.steps = 0
        self.raw_state = np.zeros((self.height, self.width), dtype=np.uint8)

        for segment in self.snake:
            self.raw_state[segment.y, segment.x] = 1

        self.raw_state[self.food.y, self.food.x] = 2

        self.steps_beyond_terminated = None

        return self._get_state(), {}

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
        if self.raw_state is None:
            return None
        color_matrix = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8)
        snake_mask = self.raw_state == 1
        food_mask = self.raw_state == 2

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

    def _get_state(self) -> np.array:
        """
        This method is called to get the state of the environment.
        Returns:
            Tuple: The state of the environment.
                Tuple[int, int]: The position of the food relative to the head of the snake.
                int: The current direction of the snake's head as an integer. (´0´ for up, ´1´ for right, ´2´ for down, ´3´ for left)
                Tuple[bool, bool, bool]: A list of 3 elements indicating if there is an obstacle in the direction of movement, to the left and to the right.
                list: A list of positions of the snake's body.
        """
        head = self.snake[0]
        tail = self.snake[-1]
        food = head - self.food
        direction = self.direction
        immediate_obstacles = [
            head + utils.DIRECTIONS[(direction + i) % 4] in self.snake for i in range(-1, 2)]

        dx_food, dy_food = food.to_tuple()

        # Flat snake, add the head and the tail
        flat_snake = [head.x, head.y, tail.x, tail.y]

        # Flatten and normalize the state
        state = [
            dx_food/self.width, dy_food/self.height, direction /
            3
        ] + immediate_obstacles + flat_snake

        return np.array(state, dtype=np.float32)


if __name__ == "__main__":
    env = SnakeEnv()
    env.reset()
    env.render()
    env.step(1)
    env.render()
    env.close()
