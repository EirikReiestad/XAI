import enum
import logging


@enum.unique
class StateType(enum.Enum):
    FULL = "full"
    PARTIAL = "partial"
    RGB = "rgb"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(state: str):
        if state == "full":
            return StateType.FULL
        if state == "partial":
            return StateType.PARTIAL
        if state == "rgb":
            return StateType.RGB
        else:
            logging.error("Invalid state type.")
            return None


@enum.unique
class Direction(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def to_tuple(self) -> tuple[int, int]:
        if self == Direction.UP:
            return (0, -1)
        if self == Direction.RIGHT:
            return (1, 0)
        if self == Direction.DOWN:
            return (0, 1)
        if self == Direction.LEFT:
            return (-1, 0)
        else:  # should never happen
            raise ValueError("Invalid direction.")


@enum.unique
class Color(enum.Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
