"""
Utility functions used for Atari2D environments.
"""

from dataclasses import dataclass


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


DIRECTIONS = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}
