from environments.gymnasium.utils import Position
from enum import Enum


class EntityType(Enum):
    OBSTACLE = 0
    BOX = 1
    AGENT = 2


class Entity:
    _position: Position
    _grabable: bool

    def __eq__(self, other):
        return self.position == other.position

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
