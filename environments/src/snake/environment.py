import pygame
import numpy as np
from ..environment import Environment
from ..grid import Grid
from .snake import Snake, SnakeSegment
from .direction import Direction
from .food import Food


class SnakeEnvironment(Environment):
    def __init__(self, name: str, description: str, width: int, height: int):
        super().__init__(name, description)
        assert isinstance(width, int), "Width must be an integer"
        assert isinstance(height, int), "Height must be an integer"
        assert width > 0, "Width must be greater than 0"
        assert height > 0, "Height must be greater than 0"

        self._rewards = {
            "move": 0,
            "eat": 1,
            "collision": -1
        }

        self.grid = Grid(width, height)
        self._init_food()
        self._init_snake()

        assert self._rewards is not None, "Rewards must not be None"
        assert self.grid is not None, "Grid must not be None"
        assert self.snake is not None, "Snake must not be None"
        assert self.food is not None, "Food must not be None"

        self.game_over = False

    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, rewards: dict):
        assert isinstance(rewards, dict), "Rewards must be a dictionary"
        assert "move" in rewards, "Reward for moving must be defined"
        assert "eat" in rewards, "Reward for eating must be defined"
        assert "collision" in rewards, "Reward for collision must be defined"
        self._rewards = rewards

    def _init_snake(self):
        starting_snake_length = 3
        assert starting_snake_length > 0, "Starting snake length must be greater than 0"
        assert self.grid.width > starting_snake_length, "Starting snake length must be less than grid width"

        head = SnakeSegment(1, self.grid.height // 2)
        self.snake = Snake(head)
        for i in range(starting_snake_length-1):
            self.snake.grow()
            self._move_snake(Direction.RIGHT)

    def _move_snake(self, direction: Direction = None) -> (bool, float):
        """
        Return:
            bool: game over
            float: reward 
        """
        if direction is None:
            direction = self.direction

        assert isinstance(
            direction, Direction), f"Direction must be a Direction, not {type(direction)}"

        if self._valid_direction(direction):
            self.direction = direction

        # Check if inside the grid
        head = self.snake.head
        new_x, new_y = head.x, head.y

        match direction:
            case Direction.UP:
                new_y -= 1
            case Direction.DOWN:
                new_y += 1
            case Direction.LEFT:
                new_x -= 1
            case Direction.RIGHT:
                new_x += 1

        game_over, reward = self._game_over(new_x, new_y)
        if game_over:
            return True, reward

        # Check if the snake eats the food
        reward = 0
        if new_x == self.food.x and new_y == self.food.y:
            self.snake.grow()
            self._init_food()
            reward += self.rewards["eat"]

        self.snake.move(self.direction)
        reward += self.rewards["move"]
        return False, reward

    def _game_over(self, new_x: int, new_y: int) -> (bool, float):
        """
        Return:
            (bool, float): game over, reward
        """
        if new_x < 0 or new_x >= self.grid.width or new_y < 0 or new_y >= self.grid.height:
            self.game_over = True
            return True, self.rewards["collision"]

        # Check if the snake collides with itself
        if self.snake.collides(new_x, new_y):
            self.game_over = True
            return True, self.rewards["collision"]

        return False, 0

    def _valid_direction(self, direction: Direction) -> bool:
        if direction is None:
            return True

        assert isinstance(
            direction, Direction), f"Direction must be a Direction, not {type(direction)}"

        # Check if the snake is moving in the opposite direction
        if direction == Direction.UP and self.snake.direction == Direction.DOWN:
            return False
        elif direction == Direction.DOWN and self.snake.direction == Direction.UP:
            return False
        elif direction == Direction.LEFT and self.snake.direction == Direction.RIGHT:
            return False
        elif direction == Direction.RIGHT and self.snake.direction == Direction.LEFT:
            return False
        return True

    def _init_food(self):
        self.food = Food()
        self.food.randomize_position(self.grid.width-1, self.grid.height-1)

    def set_screen(self, screen: pygame.Surface):
        assert screen is not None, "Screen must not be None"
        self.screen = screen
        pygame.display.set_caption(self.name)

    def run(self):
        pass

    def reset(self):
        self._init_snake()
        self._init_food()

    def render(self, screen: pygame.Surface):
        assert screen is not None, "Screen must not be None"
        # Add rendering logic here, e.g., drawing shapes or images
        cell_width = screen.get_width() / self.grid.width
        cell_height = screen.get_height() / self.grid.height

        self.grid.render(screen, cell_width, cell_height)
        self.snake.render(screen, cell_width, cell_height)
        self.food.render(screen, cell_width, cell_height)

    def step(self, action=None) -> (bool, float):
        """
        Action is per now a synonym for direction, but it can be extended to include more actions

        Parameters:
            action: Direction

        Return:
            bool: Game over
            float: Reward
        """
        # We will just have a text to direction mapping to not make it mandatory to use the Direction enum
        if isinstance(action, str):
            action = self._text_to_direction(action)

        if isinstance(action, int):
            action = self._int_to_direction(action)

        if isinstance(action, np.int64):
            action = self._int_to_direction(action.item())

        if not self._valid_direction(action):
            action = None

        if action is not None:
            assert isinstance(
                action, Direction), f"Action must be a Direction, not {type(action)}"
            return self._move_snake(action)
        else:
            return self._move_snake()

    def _text_to_direction(self, text: str) -> Direction:
        match text:
            case "UP":
                return Direction.UP
            case "DOWN":
                return Direction.DOWN
            case "LEFT":
                return Direction.LEFT
            case "RIGHT":
                return Direction.RIGHT
            case _:
                assert ValueError(text), f"Invalid direction: {text}"

    def _int_to_direction(self, number: int) -> Direction:
        direction = Direction(number)
        if direction in Direction:
            return direction

    def state_to_index(self) -> int:
        num_cells = self.grid.width * self.grid.height
        num_actions = len(Direction)
        state_index = (self.snake.head.x * self.grid.width + self.snake.head.y) * num_actions + \
            self.snake.direction * num_cells + self.food.x * self.grid.width + self.food.y

        return state_index
