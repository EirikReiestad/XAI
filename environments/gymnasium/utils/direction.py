import enum
import logging


@enum.unique
class Direction(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def tuple(self) -> tuple[int, int]:
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
