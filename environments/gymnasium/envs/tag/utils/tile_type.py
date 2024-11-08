import enum


@enum.unique
class TileType(enum.Enum):
    EMPTY = 0
    OBSTACLE = 1
    SEEKER = 2
    HIDER = 3
    BOX = 4
    POWERUP0 = 5
    POWERUP1 = 6

    def __str__(self) -> str:
        if self == TileType.EMPTY:
            return "Empty"
        if self == TileType.OBSTACLE:
            return "Obstacle"
        if self == TileType.SEEKER:
            return "Seeker"
        if self == TileType.HIDER:
            return "Hider"
        if self == TileType.BOX:
            return "Box"
        if self == TileType.POWERUP0:
            return "Powerup0"
        if self == TileType.POWERUP1:
            return "Powerup1"
        return "Unknown"

    def __add__(self, other):
        new_value = (self.value + other) % len(TileType)
        return TileType(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % len(TileType)
        return TileType(new_value)
