import pygame
from src.environment import Environment
from src.grid import Grid
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

        self.grid = Grid(width, height)
        self.direction = None
        self._init_snake()
        self._init_food()

    def _init_snake(self):
        starting_snake_length = 3
        assert starting_snake_length > 0, "Starting snake length must be greater than 0"
        assert self.grid.width > starting_snake_length, "Starting snake length must be less than grid width"

        head = SnakeSegment(1, self.grid.height // 2)
        self.snake = Snake(head)
        for i in range(starting_snake_length-1):
            self.snake.grow()
            self.move_snake(Direction.RIGHT)

    def move_snake(self, direction: Direction) -> bool:
        """
        Return:
            - bool: True if valid move, False otherwise
        """
        assert isinstance(
            direction, Direction), "Direction must be a Direction"

        # Check if the snake is moving in the opposite direction
        if self.direction is not None:
            if direction == Direction.UP and self.direction == Direction.DOWN:
                pass
            elif direction == Direction.DOWN and self.direction == Direction.UP:
                pass
            elif direction == Direction.LEFT and self.direction == Direction.RIGHT:
                pass
            elif direction == Direction.RIGHT and self.direction == Direction.LEFT:
                pass
            else:  # Valid move
                self.direction = direction
                return True

        # Check if inside the grid
        head = self.snake.head
        new_x, new_y = head.x, head.y

        match direction:

        self.snake.move(self.direction)

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
        pass

    def render(self, screen: pygame.Surface):
        assert screen is not None, "Screen must not be None"
        # Add rendering logic here, e.g., drawing shapes or images
        cell_width = screen.get_width() / self.grid.width
        cell_height = screen.get_height() / self.grid.height

        self.grid.render(screen, cell_width, cell_height)
        self.snake.render(screen, cell_width, cell_height)
        self.food.render(screen, cell_width, cell_height)

    def step(self):
        pass
