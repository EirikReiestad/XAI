from environments.gymnasium.utils import Position
from enum import Enum


class ObjectType(Enum):
    OBSTACLE = 0
    BOX = 1


class Object:
    object_type: ObjectType
    _position: Position
    _grabable: bool

    def __init__(
        self, object_type: ObjectType, position: Position, grabable: bool = False
    ):
        self.object_type = object_type
        self._position = position
        self._grabable = grabable
        self._grabbed = False

    def __eq__(self, other):
        return self.position == other.position

    @property
    def grabbed(self):
        return self._grabbed

    @grabbed.setter
    def grabbed(self, grabbed: bool):
        self._grabbed = grabbed

    @property
    def grabable(self):
        return self._grabable

    @grabable.setter
    def grabable(self, grabable: bool):
        self._grabable = grabable

    @property
    def position(self) -> Position:
        return self._position

    @position.setter
    def position(self, position: Position):
        self._position = position

    @property
    def state(self) -> list:
        return [*self.position.tuple, self.grabable, self.grabbed]

    @property
    def state_size(self) -> int:
        return 4

    @staticmethod
    def state_size_static() -> int:
        return 4


class Objects:
    def __init__(self, obstacles: list[Object], boxes: list[Object]):
        self.obstacles = obstacles
        self.boxes = boxes

    @property
    def state(self):
        states = []
        for obstacle in self.obstacles:
            states += obstacle.state
        for box in self.boxes:
            states += box.state
        return states

    @property
    def state_size(self):
        return sum([obstacle.state_size for obstacle in self.obstacles]) + sum(
            [box.state_size for box in self.boxes]
        )
