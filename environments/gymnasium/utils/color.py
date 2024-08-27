import enum


@enum.unique
class Color(enum.Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    PURPLE = (128, 0, 128)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    def __str__(self):
        return f"Color({self.name}, RGB{self.value})"
