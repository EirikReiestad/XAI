import enum
from environments.gymnasium.utils import Direction


@enum.unique
class ActionType(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    GRAB = 4
    RELEASE = 5

    @property
    def direction(self) -> Direction:
        match self:
            case ActionType.UP:
                return Direction.UP
            case ActionType.DOWN:
                return Direction.DOWN
            case ActionType.LEFT:
                return Direction.LEFT
            case ActionType.RIGHT:
                return Direction.RIGHT
            case _:
                return Direction.NONE
