import enum


@enum.unique
class DrawMode(enum.Enum):
    NOTHING = 0
    OBSTACLE = 1
    ERASE = 2
    AGENT0 = 3
    AGENT1 = 4

    def __str__(self) -> str:
        if self == DrawMode.NOTHING:
            return "Nothing"
        if self == DrawMode.OBSTACLE:
            return "Obstacle"
        if self == DrawMode.ERASE:
            return "Erase"
        return "Unknown"

    def __add__(self, other):
        new_value = (self.value + other) % (len(DrawMode) - 2)
        return DrawMode(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % (len(DrawMode) - 2)
        return DrawMode(new_value)
