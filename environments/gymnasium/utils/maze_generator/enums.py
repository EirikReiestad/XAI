import enum


@enum.unique
class TileType(enum.Enum):
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    END = 3

    def __str__(self):
        if self == TileType.EMPTY:
            return "Empty"
        if self == TileType.OBSTACLE:
            return "Obstacle"
        if self == TileType.START:
            return "Start"
        if self == TileType.END:
            return "End"

    def __add__(self, other):
        new_value = (self.value + other) % len(TileType)
        return TileType(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % len(TileType)
        return TileType(new_value)


@enum.unique
class DrawMode(enum.Enum):
    NOTHING = 0
    OBSTACLE = 1
    ERASE = 2
    START = 3
    END = 4

    def __str__(self):
        if self == DrawMode.NOTHING:
            return "Nothing"
        if self == DrawMode.OBSTACLE:
            return "Obstacle"
        if self == DrawMode.ERASE:
            return "Erase"

    def __add__(self, other):
        new_value = (self.value + other) % (len(DrawMode)-2)
        return DrawMode(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % (len(DrawMode)-2)
        return DrawMode(new_value)


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
