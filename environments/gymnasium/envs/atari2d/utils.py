"""
Utility functions used for Atari2D environments.
"""

from dataclasses import dataclass
import random


@dataclass
class Position:
    x: float | int
    y: float | int

    def __init__(self, x_or_tuple, y=None):
        if y is None:  # Assume x_or_tuple is a tuple like (x, y)
            if not isinstance(x_or_tuple, tuple) or len(x_or_tuple) != 2:
                raise ValueError(
                    "Position should be initialized with two numbers.")
            x, y = x_or_tuple
        else:
            x = x_or_tuple

        self.x = x
        self.y = y

    def __post_init__(self):
        if not isinstance(self.x, (float, int)) or not isinstance(self.y, (float, int)):
            raise ValueError(
                "Position should be initialized with two numbers.")

    def __add__(self, other):
        """Add a tuple(dx, dy) to the Position to move it."""
        if not isinstance(other, tuple) or len(other) != 2:
            raise ValueError(
                "Addition should be performed with a tuple of length 2.")

        dx, dy = other
        return Position(self.x + dx, self.y + dy)

    def __sub__(self, other):
        """Subtract a tuple(dx, dy) from the Position to move it."""
        if not isinstance(other, Position) and (not isinstance(other, tuple) or len(other) != 2):
            raise ValueError(
                "Subtraction should be performed with a tuple of length 2.")
        if isinstance(other, Position):
            dx, dy = other.x, other.y
        else:
            dx, dy = other
        return Position(self.x - dx, self.y - dy)

    def __len__(self):
        return 2

    def to_tuple(self):
        return (self.x, self.y)


DIRECTIONS = {
    0: (0, 1),  # Up
    1: (1, 0),  # Right
    2: (0, -1),  # Down
    3: (-1, 0),  # Left
}


def generate_random_position(width: int, height: int, other: [Position] = None) -> Position:
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("Width and height should be integers.")
    count = 0
    position = Position(random.randint(0, width-1),
                        random.randint(0, height-1))
    if other is None:
        return position

    while position in other:
        count += 1
        if count > width * height:
            raise ValueError(
                "Could not generate a random position.")
        position = Position(random.randint(0, width-1),
                            random.randint(0, height-1))
    return position
