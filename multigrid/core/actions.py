import enum


class Action(enum.IntEnum):
    left = 0  #: Turn left
    right = enum.auto()  #: Turn right
    forward = enum.auto()  #: Move forward
