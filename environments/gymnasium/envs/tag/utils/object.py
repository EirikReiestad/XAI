from environments.gymnasium.utils import Position
from enum import Enum


class ObjectType(Enum):
    OBSTACLE = 0
    BOX = 1
    POWERUP0 = 2
    POWERUP1 = 3


class Object:
    object_type: ObjectType
    _position: Position
    _grabable: bool

    def __init__(
        self,
        object_type: ObjectType,
        position: Position,
        grabable: bool = False,
        consumeable: bool = False,
    ):
        self.object_type = object_type
        self._position = position
        self._grabable = grabable
        self._grabbed = False
        self._consumeable = consumeable
        self._consumed = False
        self._next_position = position

    def __eq__(self, other):
        return self._position == other._position

    def can_grab(self) -> bool:
        if self._grabable and not self._grabbed:
            return True
        return False

    def can_consume(self) -> bool:
        if self._consumeable and not self._consumed:
            return True
        return False

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
    def consumeable(self):
        return self._consumeable

    @consumeable.setter
    def consumeable(self, consumeable: bool):
        self._consumeable = consumeable

    @property
    def position(self) -> Position:
        return self._position

    @position.setter
    def position(self, position: Position):
        self._position = position

    @property
    def next_position(self) -> Position:
        return self._next_position

    @next_position.setter
    def next_position(self, next_position: Position):
        self._next_position = next_position

    @property
    def state(self) -> list:
        return [*self._position.tuple, self._grabable, self._grabbed]

    @property
    def state_size(self) -> int:
        return 4

    @staticmethod
    def state_size_static() -> int:
        return 4

    @staticmethod
    def feature_names_static() -> list[str]:
        return ["x", "y", "grabable", "grabbed"]


def create_object(object_type: ObjectType, position: Position) -> Object:
    if object_type == ObjectType.OBSTACLE:
        return Object(
            object_type=ObjectType.OBSTACLE,
            position=position,
            grabable=False,
            consumeable=False,
        )
    if object_type == ObjectType.BOX:
        return Object(
            object_type=ObjectType.BOX,
            position=position,
            grabable=True,
            consumeable=False,
        )
    if object_type == ObjectType.POWERUP0:
        return Object(
            object_type=ObjectType.POWERUP0,
            position=position,
            grabable=False,
            consumeable=True,
        )
    if object_type == ObjectType.POWERUP1:
        return Object(
            object_type=ObjectType.POWERUP1,
            position=position,
            grabable=False,
            consumeable=True,
        )


class Objects:
    def __init__(
        self,
        obstacles: list[Object],
        boxes: list[Object],
        powerups0: list[Object],
        powerups1: list[Object],
    ):
        self.obstacles = obstacles
        self.boxes = boxes
        self.powerups0 = powerups0
        self.powerups1 = powerups1

    @property
    def objects(self):
        return self.obstacles + self.boxes + self.powerups0 + self.powerups1

    @property
    def state(self):
        states = []
        for obstacle in self.obstacles:
            states += obstacle.state
        for box in self.boxes:
            states += box.state
        for powerup0 in self.powerups0:
            states += powerup0.state
        for powerup1 in self.powerups1:
            states += powerup1.state
        return states

    @property
    def state_size(self):
        total_size = 0
        total_size += sum([obstacle.state_size for obstacle in self.obstacles])
        total_size += sum([box.state_size for box in self.boxes])
        total_size += sum([powerup0.state_size for powerup0 in self.powerups0])
        total_size += sum([powerup1.state_size for powerup1 in self.powerups1])
        return total_size
