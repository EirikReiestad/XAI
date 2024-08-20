import enum


@enum.unique
class TileType(enum.Enum):
    EMPTY = 0
    OBSTACLE = 1
    AGENT0 = 2
    AGENT1 = 3

    def __str__(self) -> str:
        if self == TileType.EMPTY:
            return "Empty"
        if self == TileType.OBSTACLE:
            return "Obstacle"
        if self == TileType.AGENT0:
            return "Agent1"
        if self == TileType.AGENT1:
            return "Agent2"
        return "Unknown"

    def __add__(self, other):
        new_value = (self.value + other) % len(TileType)
        return TileType(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % len(TileType)
        return TileType(new_value)
