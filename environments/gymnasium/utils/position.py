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
