import enum


@enum.unique
class MazeDrawMode(enum.Enum):
    NOTHING = 0
    OBSTACLE = 1
    ERASE = 2
    START = 3
    END = 4

    def __str__(self) -> str:
        if self == MazeDrawMode.NOTHING:
            return "Nothing"
        if self == MazeDrawMode.OBSTACLE:
            return "Obstacle"
        if self == MazeDrawMode.ERASE:
            return "Erase"
        return "Unknown"

    def __add__(self, other):
        new_value = (self.value + other) % (len(MazeDrawMode) - 2)
        return MazeDrawMode(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % (len(MazeDrawMode) - 2)
        return MazeDrawMode(new_value)
