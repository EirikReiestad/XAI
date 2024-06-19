
Utility functions used for Atari2D environments.
"""

from dataclasses import dataclass


@dataclass
class Position:
    x: float | int
    y: float | int

    def __post_init__(self):
        if self.x < 0 or self.y < 0:
            raise ValueError("Position cannot be negative.")

    def __add__(self, other):
        """Add a tuple(dx, dy) to the Position to move it."""
        if not isinstance(other, tuple) or len(other) != 2:
            raise ValueError(
                "Addition should be performed with a tuple of length 2.")

        dx, dy = other
        return Position(self.x + dx, self.y + dy)
