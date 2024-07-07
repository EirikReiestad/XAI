import enum


@enum.unique
class MazeTileType(enum.Enum):
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    END = 3

    def __str__(self):
        if self == MazeTileType.EMPTY:
            return "Empty"
        if self == MazeTileType.OBSTACLE:
            return "Obstacle"
        if self == MazeTileType.START:
            return "Start"
        if self == MazeTileType.END:
            return "End"

    def __add__(self, other):
        new_value = (self.value + other) % len(MazeTileType)
        return MazeTileType(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % len(MazeTileType)
        return MazeTileType(new_value)


@enum.unique
class MazeDrawMode(enum.Enum):
    NOTHING = 0
    OBSTACLE = 1
    ERASE = 2
    START = 3
    END = 4

    def __str__(self):
        if self == MazeDrawMode.NOTHING:
            return "Nothing"
        if self == MazeDrawMode.OBSTACLE:
            return "Obstacle"
        if self == MazeDrawMode.ERASE:
            return "Erase"

    def __add__(self, other):
        new_value = (self.value + other) % (len(MazeDrawMode)-2)
        return MazeDrawMode(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % (len(MazeDrawMode)-2)
        return MazeDrawMode(new_value)


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


class Color(enum.Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
