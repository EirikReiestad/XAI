import enum


@enum.unique
class Direction(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def to_tuple(self):
        if self == Direction.UP:
            return (0, -1)
        if self == Direction.RIGHT:
            return (1, 0)
        if self == Direction.DOWN:
            return (0, 1)
        if self == Direction.LEFT:
            return (-1, 0)


@enum.unique
class Color(enum.Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
