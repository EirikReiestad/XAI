import enum
from environments.gymnasium.envs.tag.utils import TileType
from utils.src.color import Color


@enum.unique
class DrawMode(enum.Enum):
    NOTHING = 0
    OBSTACLE = 1
    BOX = 2
    ERASE = 3
    SEEKER = 4
    HIDER = 5

    def __str__(self) -> str:
        if self == DrawMode.NOTHING:
            return "Nothing"
        if self == DrawMode.OBSTACLE:
            return "Obstacle"
        if self == DrawMode.BOX:
            return "Box"
        if self == DrawMode.ERASE:
            return "Erase"
        return "Unknown"

    def __add__(self, other):
        new_value = (self.value + other) % (len(DrawMode) - 2)
        return DrawMode(new_value)

    def __iadd__(self, other):
        new_value = (self.value + other) % (len(DrawMode) - 2)
        return DrawMode(new_value)

    @property
    def color(self) -> tuple[int, int, int]:
        return {
            DrawMode.NOTHING: Color.WHITE.value,
            DrawMode.OBSTACLE: Color.BLACK.value,
            DrawMode.BOX: Color.YELLOW.value,
            DrawMode.SEEKER: Color.BLUE.value,
            DrawMode.HIDER: Color.GREEN.value,
        }.get(self, (255, 255, 255))

    def get_highlight_color(self) -> tuple[int, int, int] | None:
        match self:
            case DrawMode.OBSTACLE:
                return (75, 75, 75)
            case DrawMode.ERASE:
                return (175, 175, 175)
            case DrawMode.SEEKER:
                return (0, 200, 0)
            case DrawMode.HIDER:
                return (200, 0, 0)
            case _:
                return None

    @staticmethod
    def from_tile_type(tile_type: TileType) -> "DrawMode":
        return {
            TileType.EMPTY: DrawMode.NOTHING,
            TileType.OBSTACLE: DrawMode.OBSTACLE,
            TileType.BOX: DrawMode.BOX,
            TileType.SEEKER: DrawMode.SEEKER,
            TileType.HIDER: DrawMode.HIDER,
        }.get(tile_type, DrawMode.NOTHING)
