import pygame
from ..direction import Direction
import numpy as np


class SnakeSegment:
    def __init__(self, x: int, y: int):
        assert isinstance(x, int), "x must be an integer"
        assert isinstance(y, int), "y must be an integer"

        self.x = x
        self.y = y

        self.next: SnakeSegment = None

    def __str__(self):
        return f"SnakeSegment({self.x}, {self.y})"

    def __copy__(self):
        return SnakeSegment(self.x, self.y)


class Snake:
    def __init__(self, segment: SnakeSegment):
        assert isinstance(segment, SnakeSegment), "head must be a SnakeSegment"
        self.head = segment
        self.should_grow = False
        self._direction = Direction.RIGHT

    def grow(self):
        self.should_grow = True

    def move(self, direction: Direction) -> bool:
        """
        Parameters:
            direction: Direction
        Return:
            bool: True if valid move, False otherwise
        """
        assert isinstance(
            direction, Direction), f"Direction must be a Direction, not {type(direction)}"

        self.direction = direction

        match direction:
            case Direction.UP:
                new_x, new_y = self.head.x, self.head.y - 1
            case Direction.DOWN:
                new_x, new_y = self.head.x, self.head.y + 1
            case Direction.LEFT:
                new_x, new_y = self.head.x - 1, self.head.y
            case Direction.RIGHT:
                new_x, new_y = self.head.x + 1, self.head.y

        new_head = SnakeSegment(new_x, new_y)
        new_head.next = self.head
        self.head = new_head

        current = self.head

        while current.next.next is not None:
            current = current.next

        if not self.should_grow:
            current.next = None

        self.should_grow = False

        return True

    def collides(self, x: int, y: int) -> bool:
        current = self.head
        while current is not None:
            if current.x == x and current.y == y:
                return True
            current = current.next
        return False

    def render(self, screen: pygame.Surface, cell_width: float, cell_height: float):
        current = self.head
        color = (0, 255, 0)  # Green
        while current is not None:
            pygame.draw.rect(screen, color, (current.x * cell_width,
                             current.y * cell_height, cell_width, cell_height))
            current = current.next

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction: Direction):
        assert isinstance(
            direction, Direction), f"Direction must be a Direction, not {type(direction)}"
        self._direction = direction

    def get_segments(self) -> np.ndarray:
        """
        Return
            np.ndarray: Array of SnakeSegment
        """
        segments = []
        current = self.head
        while current is not None:
            segments.append(current)
            current = current.next
        assert len(segments) > 0, "Snake has no segments"
        assert all(isinstance(segment, SnakeSegment)
                   for segment in segments), "All segments must be SnakeSegment"
        return segments

    def __str__(self):
        current = self.head
        result = ""
        while current is not None:
            result += f"{current} -> "
            current = current.next
        return result

    def __len__(self):
        length = 0
        current = self.head
        while current is not None:
            length += 1
            current = current.next
        return length
