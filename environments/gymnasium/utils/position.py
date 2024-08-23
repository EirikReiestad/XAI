from dataclasses import dataclass
from math import sqrt
from typing import Tuple, Union


@dataclass
class Position:
    x: Union[float, int]
    y: Union[float, int]

    def __init__(
        self,
        x: Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
        y: Union[float, int] | None = None,
    ):
        """
        Initialize a Position object. Can take either two separate coordinates or a tuple of coordinates.

        Args:
            x_or_tuple (Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]]): X-coordinate or a tuple containing (x, y).
            y (Union[float, int], optional): Y-coordinate if x_or_tuple is a single value.
        """
        if y is None:
            if isinstance(x, tuple) and len(x) == 2:
                x, y = x
            else:
                raise ValueError(
                    "When providing a single argument, it must be a tuple (x, y)."
                )
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __post_init__(self):
        if not isinstance(self.x, (float, int)) or not isinstance(self.y, (float, int)):
            raise ValueError("Coordinates must be either float or int.")

    def __add__(
        self, other: Tuple[Union[float, int], Union[float, int]] | "Position"
    ) -> "Position":
        """
        Add a tuple (dx, dy) to the Position to move it.

        Args:
            other (Tuple[Union[float, int], Union[float, int]]): The delta (dx, dy).

        Returns:
            Position: New Position after addition.
        """
        if isinstance(other, tuple):
            dx, dy = other
        elif isinstance(other, Position):
            dx, dy = other.x, other.y

        return Position(self.x + dx, self.y + dy)

    def __sub__(
        self, other: Union["Position", Tuple[Union[float, int], Union[float, int]]]
    ) -> "Position":
        """
        Subtract a tuple (dx, dy) or another Position from the Position to move it.

        Args:
            other (Union[Position, Tuple[Union[float, int], Union[float, int]]]): The delta (dx, dy) or another Position.

        Returns:
            Position: New Position after subtraction.
        """
        if isinstance(other, Position):
            dx, dy = other.x, other.y
            return Position(self.x - dx, self.y - dy)
        elif isinstance(other, tuple) and len(other) == 2:
            dx, dy = other
            return Position(self.x - dx, self.y - dy)

    def __len__(self) -> int:
        """
        Get the number of coordinates in the Position.

        Returns:
            int: Always returns 2.
        """
        return 2

    def __iter__(self):
        """
        Iterate over the coordinates.
        Yields:
            Union[float, int]: The x and y coordinates.
        """
        yield self.x
        yield self.y

    def distance_to(self, other: "Position") -> float:
        """
        Calculate the Euclidean distance between this Position and another.

        Args:
            other (Position): The other position to calculate the distance to.

        Returns:
            float: The Euclidean distance between the two positions.
        """
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @property
    def tuple(self) -> Tuple[Union[float, int], Union[float, int]]:
        """
        Convert the Position to a tuple.

        Returns:
            Tuple[Union[float, int], Union[float, int]]: The (x, y) coordinates.
        """
        return (self.x, self.y)

    @property
    def row_major_order(self) -> Tuple[int, int]:
        """
        Get the coordinates in row-major order.
        Returns:
            Tuple[int, int]: The coordinates in row-major order.
        """
        return (int(self.y), int(self.x))
