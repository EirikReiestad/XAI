import enum


@enum.unique
class MazeTileType(enum.Enum):
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    END = 3

    def __str__(self) -> str:
        if self == MazeTileType.EMPTY:
            return "Empty"
        if self == MazeTileType.OBSTACLE:
            return "Obstacle"
        if self == MazeTileType.START:
            return "Start"
        if self == MazeTileType.END:
            return "End"
        return "Unknown"

    def __add__(self, other):
        new_value = (self.value + other) % len(MazeTileType)
        return MazeTileType(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % len(MazeTileType)
        return MazeTileType(new_value)
