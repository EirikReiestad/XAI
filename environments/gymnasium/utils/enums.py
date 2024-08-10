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
    def from_string(state: str) -> "StateType":
        """
        Convert a string to a StateType enum value.

        Args:
            state (str): The state type as a string.

        Returns:
            StateType: Corresponding StateType enum value.

        Raises:
            ValueError: If the string does not match any StateType.
        """
        try:
            return StateType(state)
        except ValueError:
            logging.error(f"Invalid state type: {state}")
            raise ValueError(f"Invalid state type: {state}")


@enum.unique
class Direction(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def to_tuple(self) -> tuple[int, int]:
        """
        Convert the direction to a (dx, dy) tuple.

        Returns:
            tuple[int, int]: The corresponding (dx, dy) tuple for the direction.

        Raises:
            ValueError: If the direction is invalid (should never happen).
        """
        direction_map = {
            Direction.UP: (0, -1),
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
        }

        direction_tuple = direction_map.get(self)
        if direction_tuple is None:
            logging.error("Invalid direction.")
            raise ValueError("Invalid direction.")
        return direction_tuple


@enum.unique
class Color(enum.Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    def __str__(self):
        return f"Color({self.name}, RGB{self.value})"
