import functools
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .agent import Agent
from .constants import Color, Type
from .utils import fill_coords, point_in_rect


class WorldObject.ctMeta(type):
    _TYPE_IDX_TO_CLASS = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        if name != "WorldObject.ct":
            type_name = class_dict.get("type_name", name.lower())
            if type_name not in set(Type):
                Type.add_item(type_name, type_name)

            meta._TYPE_IDX_TO_CLASS[Type(type_name).to_index()] = cls
        return cls


class WorldObject.ct(np.ndarray, metaclass=WorldObject.ctMeta):
    TYPE = 0
    COLOR = 1
    STATE = 2

    dim = len([TYPE, COLOR, STATE])

    def __new__(cls, type: str | None = None, color: str = Color.from_index(0)):
        type_name = type or getattr(cls, "type_name", cls.__name__.lower())
        type_idx = Type(type_name).to_index()

        cls = WorldObject.ctMeta._TYPE_IDX_TO_CLASS.get(type_idx, cls)

        obj = np.zeros(cls.dim, dtype=object).view(cls)
        obj[WorldObject.ct.TYPE] = type_idx
        obj[WorldObject.ct.COLOR] = Color(color).to_index()
        obj.contains: WorldObject.ct | None = None
        obj.init_pos: tuple[int, int] | None = None
        obj.current_pos: tuple[int, int] | None = None

        return obj

    def __bool__(self):
        return self.type != Type.empty

    def __repr__(self):
        return f"{self.__class__.__name__}(color={self.color})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other: Any):
        return self.type == other.type and self.color == other.color

    @staticmethod
    @functools.cache
    def empty() -> "WorldObject.ct":
        """
        Return an empty WorldObject.instance.
        """
        return WorldObject.ct(type=Type.empty)

    @staticmethod
    def from_array(arr: ArrayLike[int]) -> "WorldObject.ct" | None:
        """
        Convert an array to a WorldObject.instance.

        Parameters
        ----------
        arr : ArrayLike[int]
            Array encoding the object type, color, and state
        """
        type_idx = arr[WorldObject.ct.TYPE]

        if type_idx == Type.empty.to_index():
            return None

        if type_idx in WorldObject.ct._TYPE_IDX_TO_CLASS:
            cls = WorldObject.ct._TYPE_IDX_TO_CLASS[type_idx]
            obj = cls.__new__(cls)
            obj[...] = arr
            return obj

        raise ValueError(f"Unknown object type: {arr[WorldObject.ct.TYPE]}")

    @functools.cached_property
    def type(self) -> Type:
        """
        Return the object type.
        """
        return Type.from_index(self[WorldObject.ct.TYPE])

    @property
    def color(self) -> Color:
        """
        Return the object color.
        """
        return Color.from_index(self[WorldObject.ct.COLOR])

    @color.setter
    def color(self, value: str):
        """
        Set the object color.
        """
        self[WorldObject.ct.COLOR] = Color(value).to_index()

    @property
    def state(self) -> str:
        """
        Return the name of the object state.
        """
        return State.from_index(self[WorldObject.STATE])

    @state.setter
    def state(self, value: str):
        """
        Set the name of the object state.
        """
        self[WorldObject.STATE] = State(value).to_index()

    def can_overlap(self) -> bool:
        """
        Can an agent overlap with this?
        """
        return self.type == Type.empty

    def can_pickup(self) -> bool:
        """
        Can an agent pick this up?
        """
        return False

    def can_contain(self) -> bool:
        """
        Can this contain another object?
        """
        return False

    def toggle(self, env: MultiGridEnv, agent: Agent, pos: tuple[int, int]) -> bool:
        """
        Toggle the state of this object or trigger an action this object performs.

        Parameters
        ----------
        env : MultiGridEnv
            The environment this object is contained in
        agent : Agent
            The agent performing the toggle action
        pos : tuple[int, int]
            The (x, y) position of this object in the environment grid

        Returns
        -------
        success : bool
            Whether the toggle action was successful
        """
        return False

    def encode(self) -> tuple[int, int, int]:
        """
        Encode a 3-tuple description of this object.

        Returns
        -------
        type_idx : int
            The index of the object type
        color_idx : int
            The index of the object color
        state_idx : int
            The index of the object state
        """
        return tuple(self)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state_idx: int) -> "WorldObject. | None:
        """
        Create an object from a 3-tuple description.

        Parameters
        ----------
        type_idx : int
            The index of the object type
        color_idx : int
            The index of the object color
        state_idx : int
            The index of the object state
        """
        arr = np.array([type_idx, color_idx, state_idx])
        return WorldObject.from_array(arr)

    def render(self, img: ndarray[np.uint8]):
        """
        Draw the world object.

        Parameters
        ----------
        img : ndarray[int] of shape (width, height, 3)
            RGB image array to render object on
        """
        raise NotImplementedError


class Goal(WorldObject.ct):
    """
    Goal object an agent may be searching for.
    """

    def __new__(cls, color: str = Color.green):
        return super().__new__(cls, color=color)

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        return True

    def render(self, img):
        """
        :meta private:
        """
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())


class Floor(WorldObject.ct):
    """
    Colored floor tile an agent can walk over.
    """

    def __new__(cls, color: str = Color.blue):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, color=color)

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        return True

    def render(self, img):
        """
        :meta private:
        """
        # Give the floor a pale color
        color = self.color.rgb() / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Wall(WorldObject.ct):
    """
    Wall object that agents cannot move through.
    """

    @functools.cache  # reuse instances, since object is effectively immutable
    def __new__(cls, color: str = Color.grey):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, color=color)

    def render(self, img):
        """
        :meta private:
        """
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())
