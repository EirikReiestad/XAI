import random
from environments.gymnasium.utils.position import Position


def generate_random_position(
    width: int, height: int, other: list[Position] | None = None
) -> Position:
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("Width and height should be integers.")
    count = 0
    position = Position(random.randint(0, width - 1), random.randint(0, height - 1))
    if other is None:
        return position

    while position in other:
        count += 1
        if count > width * height:
            raise ValueError("Could not generate a random position.")
        position = Position(random.randint(0, width - 1), random.randint(0, height - 1))
    return position
