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
            self._move_snake(Direction.RIGHT)

    def _move_snake(self, direction: Direction = None) -> bool:
        """
        Return:
            - bool: True if valid move, False otherwise
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

        if new_x < 0 or new_x >= self.grid.width or new_y < 0 or new_y >= self.grid.height:
            False

        return self.snake.move(self.direction)

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
        pass

    def render(self, screen: pygame.Surface):
        assert screen is not None, "Screen must not be None"
        # Add rendering logic here, e.g., drawing shapes or images
        cell_width = screen.get_width() / self.grid.width
        cell_height = screen.get_height() / self.grid.height

        self.grid.render(screen, cell_width, cell_height)
        self.snake.render(screen, cell_width, cell_height)
        self.food.render(screen, cell_width, cell_height)

    def step(self, action=None):
        """
        Action is per now a synonym for direction, but it can be extended to include more actions

        Parameters:
            action: Direction
        """
        # We will just have a text to direction mapping to not make it mandatory to use the Direction enum
        if isinstance(action, str):
            action = self._text_to_direction(action)

        if not self._valid_direction(action):
            action = None

        if action is not None:
            self._move_snake(action)
        else:
            self._move_snake()

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
