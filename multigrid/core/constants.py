import enum

import numpy as np
from numpy.typing import NDArray as ndarray
from ..utils.enum import IndexedEnum

TILE_PIXELS = 32

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
}


DIRECTION_TO_VECTOR = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class Color(str, IndexedEnum):
    """
    Enumeration of object colors.
    """

    red = "red"
    green = "green"
    blue = "blue"
    purple = "purple"
    yellow = "yellow"
    grey = "grey"

    @classmethod
    def add_color(cls, name: str, rgb: ndarray[np.uint8]):
        """
        Add a new color to the ``Color`` enumeration.

        Parameters
        ----------
        name : str
            Name of the new color
        rgb : ndarray[np.uint8] of shape (3,)
            RGB value of the new color
        """
        cls.add_item(name, name)
        COLORS[name] = np.asarray(rgb, dtype=np.uint8)

    @staticmethod
    def cycle(n: int) -> tuple["Color", ...]:
        """
        Return a cycle of ``n`` colors.
        """
        return tuple(Color.from_index(i % len(Color)) for i in range(int(n)))

    def rgb(self) -> ndarray[np.uint8]:
        """
        Return the RGB value of this ``Color``.
        """
        return COLORS[self]


class Direction(enum.IntEnum):
    """
    Enumeration of agent directions.
    """

    right = 0
    down = 1
    left = 2
    up = 3

    def to_vec(self) -> ndarray[np.int8]:
        """
        Return the vector corresponding to this ``Direction``.
        """
        return DIRECTION_TO_VECTOR[self]


class Type(str, IndexedEnum):
    """
    Enumeration of object types.
    """

    unseen = "unseen"
    empty = "empty"
    wall = "wall"
    floor = "floor"
    door = "door"
    key = "key"
    ball = "ball"
    box = "box"
    goal = "goal"
    lava = "lava"
    agent = "agent"
